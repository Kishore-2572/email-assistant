# Email Generation Assistant

An AI-powered email generation assistant that produces professional emails based on user input, with a built-in evaluation framework for measuring output quality using custom metrics.

## Features

- **Email Generation**: Generate professional emails from Intent, Key Facts, and Tone inputs
- **Two Prompting Strategies**: Advanced (Few-Shot + Role-Playing) vs Baseline (Zero-Shot) for comparison
- **Custom Evaluation Metrics**: Three tailored metrics — Fact Recall, Tone Accuracy, Conciseness & Clarity
- **Evaluation Pipeline**: Automated testing across 10 diverse scenarios with JSON/CSV report output
- **Comparative Analysis**: Data-driven comparison of prompting strategies with production recommendation
- **Interactive Web UI**: Streamlit interface for real-time email generation

## Prompt Engineering Approach

### Strategy A: Advanced (Few-Shot + Role-Playing)

Combines two advanced prompting techniques:

1. **Role-Playing**: The model is assigned the persona of a "senior executive communications specialist with 15 years of experience" to establish expertise and quality expectations.
2. **Few-Shot Examples**: Two diverse email examples (formal meeting follow-up + casual team update) demonstrate the expected output format and quality, helping the model understand the task through demonstration.

### Strategy B: Baseline (Zero-Shot)

A minimal prompt with a simple system instruction ("Write a professional email based on the user's request") and no examples, serving as a control for measuring the impact of advanced prompting.

## Custom Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **Fact Recall** | LLM-as-a-Judge | Measures whether all key facts from the input are present and accurately represented. Evaluates presence, accuracy, and natural integration. |
| **Tone Accuracy** | LLM-as-a-Judge | Measures how well the email matches the requested tone using tone-specific indicators for vocabulary, structure, and emotional register. |
| **Conciseness & Clarity** | Hybrid (30% automated + 70% LLM judge) | Automated word-count ratio analysis (optimal 0.8-1.2x reference length) combined with LLM assessment of filler, redundancy, and readability. |

All metrics score on a 0-10 scale.

## Setup

### Prerequisites

- Python 3.11 or higher
- OpenAI API key with access to GPT-4o

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd email-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Run the Streamlit Web UI

```bash
streamlit run src/app.py
```

Opens a browser with the interactive email generation interface. Select a strategy and tone, enter your intent and key facts, and generate emails.

### Run the Evaluation Pipeline

```bash
python -m src.evaluation.evaluator
```

Runs all 10 test scenarios through both strategies, evaluates with 3 custom metrics, and generates reports in the `reports/` directory:

- `reports/evaluation_results.json` — Complete structured results
- `reports/evaluation_results.csv` — Tabular results for spreadsheet analysis
- `reports/comparative_analysis.md` — Strategy comparison and recommendation

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
email-assistant/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration and environment loading
│   ├── email_generator.py         # Core email generation logic
│   ├── prompt_templates.py        # Prompt engineering strategies
│   ├── app.py                     # Streamlit web UI
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py             # 3 custom metric implementations
│       ├── evaluator.py           # Evaluation pipeline orchestrator
│       └── test_scenarios.py      # 10 test scenarios with reference emails
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_email_generator.py
│   ├── test_prompt_templates.py
│   └── test_metrics.py
├── reports/                       # Generated evaluation output
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Test Scenarios

The evaluation uses 10 diverse scenarios covering:

| # | Scenario | Tone |
|---|----------|------|
| 1 | Meeting Follow-Up | Formal |
| 2 | Team Update | Casual |
| 3 | Deadline Reminder | Urgent |
| 4 | Customer Apology | Apologetic |
| 5 | Partnership Proposal | Persuasive |
| 6 | Job Application Follow-Up | Formal |
| 7 | Event Invitation | Casual |
| 8 | Project Delay Notification | Empathetic |
| 9 | Request for Proposal Details | Formal |
| 10 | Policy Change Announcement | Persuasive |
