"""Evaluation pipeline orchestrator.

Runs all 10 scenarios through both strategies, evaluates with 3 metrics,
and generates JSON, CSV, and comparative analysis reports.
"""

import csv
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.config import OPENAI_MODEL, STRATEGIES, logger, validate_api_key
from src.email_generator import EmailInput, GeneratedEmail, generate_email
from src.evaluation.metrics import (
    METRIC_DEFINITIONS,
    MetricScore,
    evaluate_conciseness_clarity,
    evaluate_fact_recall,
    evaluate_tone_accuracy,
)
from src.evaluation.test_scenarios import TestScenario, get_all_scenarios


@dataclass
class EvaluationResult:
    """Complete evaluation result for one scenario + strategy."""

    scenario_id: int
    scenario_name: str
    strategy: str
    generated_subject: str
    generated_body: str
    scores: list[MetricScore]
    average_score: float


@dataclass
class StrategyStats:
    """Aggregate statistics for a strategy."""

    strategy: str
    average_score: float
    per_metric_average: dict[str, float]
    best_scenario_id: int
    best_scenario_score: float
    worst_scenario_id: int
    worst_scenario_score: float


def _evaluate_single(
    scenario: TestScenario,
    generated: GeneratedEmail,
) -> list[MetricScore]:
    """Evaluate a single generated email with all 3 metrics.

    Args:
        scenario: The test scenario with reference email.
        generated: The generated email to evaluate.

    Returns:
        List of 3 MetricScore objects.
    """
    scores: list[MetricScore] = []

    inp = scenario.input
    ref = scenario.reference

    # Metric 1: Fact Recall
    try:
        fact_score = evaluate_fact_recall(
            intent=inp.intent,
            key_facts=inp.key_facts,
            tone=inp.tone,
            gen_subject=generated.subject,
            gen_body=generated.body,
            ref_subject=ref.subject,
            ref_body=ref.body,
        )
        scores.append(fact_score)
    except RuntimeError as e:
        logger.error("Fact recall evaluation failed for scenario %d | %s", scenario.id, str(e))
        scores.append(MetricScore("fact_recall", 0.0, f"Evaluation failed: {e}"))

    # Metric 2: Tone Accuracy
    try:
        tone_score = evaluate_tone_accuracy(
            intent=inp.intent,
            tone=inp.tone,
            gen_subject=generated.subject,
            gen_body=generated.body,
            ref_subject=ref.subject,
            ref_body=ref.body,
        )
        scores.append(tone_score)
    except RuntimeError as e:
        logger.error("Tone accuracy evaluation failed for scenario %d | %s", scenario.id, str(e))
        scores.append(MetricScore("tone_accuracy", 0.0, f"Evaluation failed: {e}"))

    # Metric 3: Conciseness & Clarity
    try:
        conciseness_score = evaluate_conciseness_clarity(
            intent=inp.intent,
            key_facts=inp.key_facts,
            gen_subject=generated.subject,
            gen_body=generated.body,
            ref_subject=ref.subject,
            ref_body=ref.body,
        )
        scores.append(conciseness_score)
    except RuntimeError as e:
        logger.error(
            "Conciseness evaluation failed for scenario %d | %s", scenario.id, str(e)
        )
        scores.append(MetricScore("conciseness_clarity", 0.0, f"Evaluation failed: {e}"))

    return scores


def run_evaluation(strategies: list[str] | None = None) -> list[EvaluationResult]:
    """Run the full evaluation pipeline.

    Args:
        strategies: List of strategies to evaluate. Defaults to all strategies.

    Returns:
        List of EvaluationResult for all scenario-strategy combinations.
    """
    validate_api_key()

    if strategies is None:
        strategies = STRATEGIES

    scenarios = get_all_scenarios()
    results: list[EvaluationResult] = []
    total = len(scenarios) * len(strategies)
    completed = 0

    # Phase 1: Generate all emails
    logger.info("=== Phase 1: Generating emails ===")
    generated_emails: dict[tuple[int, str], GeneratedEmail] = {}

    for scenario in scenarios:
        for strategy in strategies:
            logger.info(
                "Generating [%d/%d] | scenario=%d (%s) | strategy=%s",
                completed + 1,
                total,
                scenario.id,
                scenario.name,
                strategy,
            )
            print(
                f"  Generating scenario {scenario.id}/{len(scenarios)} "
                f"({scenario.name}) with {strategy} strategy..."
            )

            email_input = EmailInput(
                intent=scenario.input.intent,
                key_facts=scenario.input.key_facts,
                tone=scenario.input.tone,
            )

            try:
                generated = generate_email(email_input, strategy)
                generated_emails[(scenario.id, strategy)] = generated
            except (ValueError, RuntimeError) as e:
                logger.error(
                    "Generation failed | scenario=%d | strategy=%s | error=%s",
                    scenario.id,
                    strategy,
                    str(e),
                )
                generated_emails[(scenario.id, strategy)] = GeneratedEmail(
                    subject="[Generation Failed]",
                    body=f"Error: {e}",
                    strategy=strategy,
                    model=OPENAI_MODEL,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            completed += 1

    # Phase 2: Evaluate all emails
    logger.info("=== Phase 2: Evaluating emails ===")
    completed = 0

    for scenario in scenarios:
        for strategy in strategies:
            completed += 1
            generated = generated_emails[(scenario.id, strategy)]

            logger.info(
                "Evaluating [%d/%d] | scenario=%d (%s) | strategy=%s",
                completed,
                total,
                scenario.id,
                scenario.name,
                strategy,
            )
            print(
                f"  Evaluating scenario {scenario.id}/{len(scenarios)} "
                f"({scenario.name}) with {strategy} strategy..."
            )

            scores = _evaluate_single(scenario, generated)
            avg = round(sum(s.score for s in scores) / len(scores), 2) if scores else 0.0

            results.append(
                EvaluationResult(
                    scenario_id=scenario.id,
                    scenario_name=scenario.name,
                    strategy=strategy,
                    generated_subject=generated.subject,
                    generated_body=generated.body,
                    scores=scores,
                    average_score=avg,
                )
            )

    logger.info("Evaluation complete | total_results=%d", len(results))
    return results


def compute_strategy_stats(results: list[EvaluationResult]) -> dict[str, StrategyStats]:
    """Compute aggregate statistics per strategy.

    Args:
        results: All evaluation results.

    Returns:
        Dict mapping strategy name to StrategyStats.
    """
    stats: dict[str, StrategyStats] = {}

    for strategy in STRATEGIES:
        strategy_results = [r for r in results if r.strategy == strategy]
        if not strategy_results:
            continue

        all_scores = [r.average_score for r in strategy_results]
        avg = round(sum(all_scores) / len(all_scores), 2)

        # Per-metric averages
        metric_totals: dict[str, list[float]] = {}
        for r in strategy_results:
            for s in r.scores:
                metric_totals.setdefault(s.metric_name, []).append(s.score)

        per_metric = {
            k: round(sum(v) / len(v), 2) for k, v in metric_totals.items()
        }

        best = max(strategy_results, key=lambda r: r.average_score)
        worst = min(strategy_results, key=lambda r: r.average_score)

        stats[strategy] = StrategyStats(
            strategy=strategy,
            average_score=avg,
            per_metric_average=per_metric,
            best_scenario_id=best.scenario_id,
            best_scenario_score=best.average_score,
            worst_scenario_id=worst.scenario_id,
            worst_scenario_score=worst.average_score,
        )

    return stats


def generate_json_report(results: list[EvaluationResult]) -> str:
    """Generate JSON evaluation report.

    Args:
        results: All evaluation results.

    Returns:
        JSON string of the complete report.
    """
    stats = compute_strategy_stats(results)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": OPENAI_MODEL,
        "strategies": STRATEGIES,
        "metric_definitions": [asdict(m) for m in METRIC_DEFINITIONS],
        "results": [
            {
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "strategy": r.strategy,
                "generated_subject": r.generated_subject,
                "generated_body": r.generated_body,
                "scores": {s.metric_name: {"score": s.score, "reasoning": s.reasoning} for s in r.scores},
                "average_score": r.average_score,
            }
            for r in results
        ],
        "summary": {
            strategy: asdict(s) for strategy, s in stats.items()
        },
    }

    return json.dumps(report, indent=2, ensure_ascii=False)


def generate_csv_report(results: list[EvaluationResult]) -> str:
    """Generate CSV evaluation report.

    Args:
        results: All evaluation results.

    Returns:
        CSV string with one row per scenario-strategy combination.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "scenario_id",
        "scenario_name",
        "strategy",
        "fact_recall",
        "tone_accuracy",
        "conciseness_clarity",
        "average_score",
    ])

    for r in results:
        score_map = {s.metric_name: s.score for s in r.scores}
        writer.writerow([
            r.scenario_id,
            r.scenario_name,
            r.strategy,
            score_map.get("fact_recall", ""),
            score_map.get("tone_accuracy", ""),
            score_map.get("conciseness_clarity", ""),
            r.average_score,
        ])

    return output.getvalue()


def generate_comparative_analysis(results: list[EvaluationResult]) -> str:
    """Generate markdown comparative analysis report.

    Args:
        results: All evaluation results.

    Returns:
        Markdown string with comparative analysis.
    """
    stats = compute_strategy_stats(results)

    if len(stats) < 2:
        return "# Comparative Analysis\n\nInsufficient data for comparison."

    advanced = stats.get("advanced")
    baseline = stats.get("baseline")

    if not advanced or not baseline:
        return "# Comparative Analysis\n\nMissing strategy data for comparison."

    winner = "advanced" if advanced.average_score >= baseline.average_score else "baseline"
    loser = "baseline" if winner == "advanced" else "advanced"
    winner_stats = stats[winner]
    loser_stats = stats[loser]

    # Find biggest failure mode
    metric_gaps: dict[str, float] = {}
    for metric_key in winner_stats.per_metric_average:
        w_score = winner_stats.per_metric_average.get(metric_key, 0)
        l_score = loser_stats.per_metric_average.get(metric_key, 0)
        metric_gaps[metric_key] = round(w_score - l_score, 2)

    biggest_gap_metric = max(metric_gaps, key=lambda k: metric_gaps[k])
    biggest_gap_value = metric_gaps[biggest_gap_metric]

    # Find worst scenarios for losing strategy
    loser_results = sorted(
        [r for r in results if r.strategy == loser],
        key=lambda r: r.average_score,
    )
    worst_3 = loser_results[:3]

    metric_name_map = {
        "fact_recall": "Fact Recall",
        "tone_accuracy": "Tone Accuracy",
        "conciseness_clarity": "Conciseness & Clarity",
    }

    strategy_label = {
        "advanced": "Advanced (Few-Shot + Role-Playing)",
        "baseline": "Baseline (Zero-Shot)",
    }

    report = f"""# Comparative Analysis: Email Generation Strategies

## Overview

This analysis compares two prompting strategies using the same model ({OPENAI_MODEL}) across 10 diverse email generation scenarios, evaluated with 3 custom metrics.

| Metric | {strategy_label['advanced']} | {strategy_label['baseline']} | Difference |
|--------|{'-' * len(strategy_label['advanced'])}--|{'-' * len(strategy_label['baseline'])}--|------------|
| **Fact Recall** | {advanced.per_metric_average.get('fact_recall', 'N/A')} | {baseline.per_metric_average.get('fact_recall', 'N/A')} | {metric_gaps.get('fact_recall', 'N/A'):+.2f} |
| **Tone Accuracy** | {advanced.per_metric_average.get('tone_accuracy', 'N/A')} | {baseline.per_metric_average.get('tone_accuracy', 'N/A')} | {metric_gaps.get('tone_accuracy', 'N/A'):+.2f} |
| **Conciseness & Clarity** | {advanced.per_metric_average.get('conciseness_clarity', 'N/A')} | {baseline.per_metric_average.get('conciseness_clarity', 'N/A')} | {metric_gaps.get('conciseness_clarity', 'N/A'):+.2f} |
| **Overall Average** | **{advanced.average_score}** | **{baseline.average_score}** | **{advanced.average_score - baseline.average_score:+.2f}** |

## 1. Which Strategy Performed Better?

**{strategy_label[winner]}** outperformed with an overall average score of **{winner_stats.average_score}/10** compared to **{loser_stats.average_score}/10** for {strategy_label[loser]}.

The winning strategy showed its best performance on Scenario {winner_stats.best_scenario_id} (score: {winner_stats.best_scenario_score}) and its weakest on Scenario {winner_stats.worst_scenario_id} (score: {winner_stats.worst_scenario_score}).

## 2. Biggest Failure Mode of the Lower-Performing Strategy

The **{strategy_label[loser]}** strategy's biggest weakness was in **{metric_name_map.get(biggest_gap_metric, biggest_gap_metric)}**, where it scored **{biggest_gap_value:.2f} points lower** than the winning strategy.

The 3 worst-performing scenarios for {strategy_label[loser]} were:
"""

    for r in worst_3:
        score_detail = ", ".join(
            f"{metric_name_map.get(s.metric_name, s.metric_name)}: {s.score}"
            for s in r.scores
        )
        report += f"- **Scenario {r.scenario_id}** ({r.scenario_name}): avg {r.average_score} ({score_detail})\n"

    report += f"""
## 3. Production Recommendation

**Recommended strategy: {strategy_label[winner]}**

**Justification based on metric data:**
- Achieved a **{abs(advanced.average_score - baseline.average_score):.2f}-point advantage** in overall score
- Strongest improvement in {metric_name_map.get(biggest_gap_metric, biggest_gap_metric)} (+{biggest_gap_value:.2f})
- Best scenario score: {winner_stats.best_scenario_score}/10 (Scenario {winner_stats.best_scenario_id})
- Worst scenario score: {winner_stats.worst_scenario_score}/10 (Scenario {winner_stats.worst_scenario_id}) — still {'acceptable' if winner_stats.worst_scenario_score >= 6.0 else 'room for improvement'}

The {'few-shot examples and role-playing context provide the model with clear quality expectations and output patterns, resulting in more consistent and higher-quality emails' if winner == 'advanced' else 'simpler prompting approach proved more effective, suggesting the model performs well with minimal instruction for this task'}.

---

*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} using {OPENAI_MODEL}*
"""

    return report


def save_reports(results: list[EvaluationResult], output_dir: str = "reports") -> None:
    """Save all evaluation reports to disk.

    Args:
        results: All evaluation results.
        output_dir: Directory to write reports to.
    """
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(generate_json_report(results))
    logger.info("JSON report saved | path=%s", json_path)
    print(f"  Saved: {json_path}")

    # CSV report
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(generate_csv_report(results))
    logger.info("CSV report saved | path=%s", csv_path)
    print(f"  Saved: {csv_path}")

    # Comparative analysis
    analysis_path = os.path.join(output_dir, "comparative_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(generate_comparative_analysis(results))
    logger.info("Comparative analysis saved | path=%s", analysis_path)
    print(f"  Saved: {analysis_path}")


def main() -> None:
    """CLI entry point for running the evaluation pipeline."""
    print("\n=== Email Generation Assistant — Evaluation Pipeline ===\n")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Scenarios: {len(get_all_scenarios())}")
    print(f"Metrics: {len(METRIC_DEFINITIONS)}")
    print()

    print("Phase 1: Generating emails...")
    print("Phase 2: Evaluating with custom metrics...")
    print()

    results = run_evaluation()

    print("\nPhase 3: Generating reports...")
    save_reports(results)

    # Print summary
    stats = compute_strategy_stats(results)
    print("\n=== Summary ===")
    for strategy, s in stats.items():
        print(f"\n  {strategy.upper()} strategy:")
        print(f"    Overall average: {s.average_score}/10")
        for metric, score in s.per_metric_average.items():
            print(f"    {metric}: {score}/10")

    if len(stats) >= 2:
        winner = max(stats.values(), key=lambda s: s.average_score)
        print(f"\n  Winner: {winner.strategy.upper()} ({winner.average_score}/10)")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
