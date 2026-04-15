"""Test scenarios for evaluation — 10 diverse email generation scenarios."""

from dataclasses import dataclass


@dataclass
class ReferenceEmail:
    """Human-written ideal email for a test scenario."""

    subject: str
    body: str


@dataclass
class EmailInputData:
    """Input data for a test scenario."""

    intent: str
    key_facts: list[str]
    tone: str


@dataclass
class TestScenario:
    """A single test case for evaluation."""

    id: int
    name: str
    input: EmailInputData
    reference: ReferenceEmail


SCENARIOS: list[TestScenario] = [
    TestScenario(
        id=1,
        name="Meeting Follow-Up (Formal)",
        input=EmailInputData(
            intent="Follow up after a client meeting to summarize action items",
            key_facts=[
                "Met with Acme Corp's VP of Engineering, Sarah Chen",
                "Discussed migrating their legacy system to cloud infrastructure",
                "Agreed on a 3-month timeline starting January 15",
                "Next meeting scheduled for December 5 at 2 PM",
            ],
            tone="formal",
        ),
        reference=ReferenceEmail(
            subject="Follow-Up: Acme Corp Cloud Migration Discussion",
            body=(
                "Dear Ms. Chen,\n\n"
                "Thank you for taking the time to meet with us today to discuss "
                "Acme Corp's cloud infrastructure migration. I wanted to summarize "
                "the key points from our conversation for alignment.\n\n"
                "We discussed the migration of your legacy system to cloud "
                "infrastructure. I am pleased to confirm that we have agreed upon "
                "a 3-month timeline, with the project commencing on January 15.\n\n"
                "Our next meeting is scheduled for December 5 at 2:00 PM, where we "
                "can review the detailed migration plan and address any preliminary "
                "questions.\n\n"
                "Please do not hesitate to reach out if you have any questions in "
                "the meantime.\n\n"
                "Best regards"
            ),
        ),
    ),
    TestScenario(
        id=2,
        name="Team Update (Casual)",
        input=EmailInputData(
            intent="Share a project milestone with the team",
            key_facts=[
                "The beta version of the mobile app launched successfully",
                "Over 500 users signed up in the first 48 hours",
                "Two critical bugs were found and fixed within 6 hours",
                "Team celebration planned for Friday at 5 PM",
            ],
            tone="casual",
        ),
        reference=ReferenceEmail(
            subject="We Did It! Beta Launch is Live",
            body=(
                "Hey team,\n\n"
                "Just wanted to share some awesome news — our mobile app beta is "
                "officially live, and it's going great!\n\n"
                "We've already had over 500 users sign up in just the first 48 "
                "hours, which is way above what we expected. We did run into two "
                "critical bugs, but the team knocked them out within 6 hours. "
                "Seriously impressive work.\n\n"
                "To celebrate this milestone, we're getting together Friday at "
                "5 PM. Drinks and snacks are on us!\n\n"
                "Thanks for all the hard work that got us here. You all rock.\n\n"
                "Cheers"
            ),
        ),
    ),
    TestScenario(
        id=3,
        name="Urgent Deadline Reminder (Urgent)",
        input=EmailInputData(
            intent="Remind the team about an approaching compliance deadline",
            key_facts=[
                "SOC 2 compliance audit is in 5 business days",
                "Three departments still have incomplete documentation",
                "Missing items: access control logs, incident response plan, vendor risk assessments",
                "Failure to comply could result in $50,000 in penalties",
            ],
            tone="urgent",
        ),
        reference=ReferenceEmail(
            subject="URGENT: SOC 2 Compliance Deadline in 5 Business Days — Action Required",
            body=(
                "Team,\n\n"
                "This is a critical reminder that our SOC 2 compliance audit is "
                "in 5 business days. We are not yet ready, and immediate action "
                "is required.\n\n"
                "Three departments still have incomplete documentation. The "
                "following items are outstanding and must be submitted immediately:\n"
                "- Access control logs\n"
                "- Incident response plan\n"
                "- Vendor risk assessments\n\n"
                "Please be aware that failure to comply could result in penalties "
                "of up to $50,000.\n\n"
                "I need all missing documentation submitted by end of day "
                "Wednesday. Please confirm receipt of this email and your expected "
                "completion date.\n\n"
                "This is our top priority.\n\n"
                "Thank you"
            ),
        ),
    ),
    TestScenario(
        id=4,
        name="Customer Apology (Apologetic)",
        input=EmailInputData(
            intent="Apologize to a customer for a service outage",
            key_facts=[
                "Platform experienced a 4-hour outage on November 20",
                "Root cause was a database failover issue",
                "2,300 customers were affected",
                "15% service credit being offered for the current billing cycle",
            ],
            tone="apologetic",
        ),
        reference=ReferenceEmail(
            subject="Our Sincere Apology for the Service Disruption on November 20",
            body=(
                "Dear Valued Customer,\n\n"
                "I am writing to sincerely apologize for the service disruption "
                "you experienced on November 20. We understand how important "
                "uninterrupted access to our platform is for your business, and "
                "we deeply regret the inconvenience caused.\n\n"
                "Our platform experienced a 4-hour outage that affected "
                "approximately 2,300 customers. After thorough investigation, we "
                "identified the root cause as a database failover issue, which our "
                "engineering team has since resolved and implemented safeguards "
                "against.\n\n"
                "To express our commitment to making this right, we are offering "
                "a 15% service credit on your current billing cycle. This credit "
                "will be applied automatically to your next invoice.\n\n"
                "We take the reliability of our service seriously, and we are "
                "committed to ensuring this does not happen again. Thank you for "
                "your patience and continued trust.\n\n"
                "Sincerely"
            ),
        ),
    ),
    TestScenario(
        id=5,
        name="Partnership Proposal (Persuasive)",
        input=EmailInputData(
            intent="Propose a partnership with a potential collaborator",
            key_facts=[
                "Our company has 2 million active users in the fitness space",
                "Their company leads in nutrition content with 50,000+ recipes",
                "Joint integration could increase engagement by 40% based on market research",
                "Proposing a 6-month pilot program with shared revenue model",
            ],
            tone="persuasive",
        ),
        reference=ReferenceEmail(
            subject="Strategic Partnership Opportunity: Fitness Meets Nutrition",
            body=(
                "Dear Partner,\n\n"
                "I am reaching out because I believe there is a compelling "
                "opportunity for our companies to create something truly valuable "
                "together.\n\n"
                "With our 2 million active users in the fitness space and your "
                "leadership in nutrition content with over 50,000 recipes, we are "
                "uniquely positioned to offer users a comprehensive health and "
                "wellness experience. Our market research indicates that a joint "
                "integration could increase user engagement by up to 40% — a "
                "significant win for both of our platforms.\n\n"
                "I would like to propose a 6-month pilot program with a shared "
                "revenue model. This low-risk approach would allow both teams to "
                "validate the partnership while delivering immediate value to our "
                "users.\n\n"
                "I would love to schedule a call to discuss this further. Are you "
                "available next week for a 30-minute conversation?\n\n"
                "Looking forward to exploring this together.\n\n"
                "Best regards"
            ),
        ),
    ),
    TestScenario(
        id=6,
        name="Job Application Follow-Up (Formal)",
        input=EmailInputData(
            intent="Follow up on a job application submitted two weeks ago",
            key_facts=[
                "Applied for Senior Software Engineer position on November 1",
                "Has 8 years of experience in distributed systems",
                "Previously led a team of 12 engineers at TechCorp",
                "Particularly excited about the company's work on real-time data pipelines",
            ],
            tone="formal",
        ),
        reference=ReferenceEmail(
            subject="Follow-Up: Senior Software Engineer Application — November 1",
            body=(
                "Dear Hiring Manager,\n\n"
                "I hope this message finds you well. I am writing to follow up on "
                "my application for the Senior Software Engineer position, which I "
                "submitted on November 1.\n\n"
                "I remain highly enthusiastic about this opportunity. With 8 years "
                "of experience in distributed systems and my previous role leading "
                "a team of 12 engineers at TechCorp, I am confident that my "
                "background aligns well with the demands of this position.\n\n"
                "I am particularly excited about your company's work on real-time "
                "data pipelines, an area where I have significant hands-on "
                "experience and deep interest.\n\n"
                "I would welcome the opportunity to discuss how my experience can "
                "contribute to your team's goals. Please let me know if there is "
                "any additional information I can provide.\n\n"
                "Thank you for your time and consideration.\n\n"
                "Respectfully"
            ),
        ),
    ),
    TestScenario(
        id=7,
        name="Event Invitation (Casual)",
        input=EmailInputData(
            intent="Invite colleagues to an upcoming team-building event",
            key_facts=[
                "Annual team offsite at Lake Tahoe on January 20-22",
                "Activities include hiking, kayaking, and a cooking class",
                "Company covers travel and accommodation",
                "RSVP needed by December 15",
            ],
            tone="casual",
        ),
        reference=ReferenceEmail(
            subject="You're Invited! Team Offsite at Lake Tahoe",
            body=(
                "Hey everyone,\n\n"
                "Exciting news — our annual team offsite is happening at Lake "
                "Tahoe from January 20 to 22! This is going to be a blast.\n\n"
                "We've got some great activities lined up, including hiking, "
                "kayaking, and even a cooking class. Something for everyone! And "
                "the best part — the company is covering all travel and "
                "accommodation, so no out-of-pocket costs for you.\n\n"
                "All I need from you is an RSVP by December 15 so we can finalize "
                "the arrangements.\n\n"
                "Hope to see everyone there!\n\n"
                "Cheers"
            ),
        ),
    ),
    TestScenario(
        id=8,
        name="Project Delay Notification (Empathetic)",
        input=EmailInputData(
            intent="Inform a client about a project delay",
            key_facts=[
                "Mobile app delivery delayed by 3 weeks",
                "Delay caused by unexpected third-party API changes",
                "New estimated delivery date is February 28",
                "Adding two extra developers to accelerate remaining work",
            ],
            tone="empathetic",
        ),
        reference=ReferenceEmail(
            subject="Update on Your Mobile App Project Timeline",
            body=(
                "Dear Client,\n\n"
                "I want to be transparent with you about an update to our project "
                "timeline. I understand how important the timely delivery of your "
                "mobile app is, and I want to address this directly.\n\n"
                "Unfortunately, we have encountered unexpected changes to a "
                "third-party API that our application integrates with, which has "
                "impacted our development schedule. As a result, we are looking at "
                "a 3-week delay, with a new estimated delivery date of "
                "February 28.\n\n"
                "I know this is not the news you were hoping for, and I sincerely "
                "understand the impact this may have on your plans. To minimize "
                "further delays, we are adding two extra developers to the team to "
                "accelerate the remaining work.\n\n"
                "I want to assure you that the quality of the final product "
                "remains our top priority. I am available to discuss this in more "
                "detail at your convenience.\n\n"
                "Thank you for your understanding and patience.\n\n"
                "Warm regards"
            ),
        ),
    ),
    TestScenario(
        id=9,
        name="Request for Proposal Details (Formal)",
        input=EmailInputData(
            intent="Request additional details about a vendor's proposal",
            key_facts=[
                "Received their initial proposal for data analytics platform",
                "Need clarification on pricing for the enterprise tier",
                "Require details on data residency and compliance certifications",
                "Decision needs to be made by December 30",
            ],
            tone="formal",
        ),
        reference=ReferenceEmail(
            subject="Request for Additional Details — Data Analytics Platform Proposal",
            body=(
                "Dear Vendor Team,\n\n"
                "Thank you for submitting your proposal for the data analytics "
                "platform. We have reviewed the initial document and are interested "
                "in moving forward. However, we require additional information "
                "before we can make our final decision.\n\n"
                "Specifically, we would appreciate clarification on the following:\n"
                "1. Pricing structure for the enterprise tier, including any "
                "volume-based discounts\n"
                "2. Data residency options and geographic availability\n"
                "3. Compliance certifications held (SOC 2, GDPR, HIPAA, etc.)\n\n"
                "Please note that we need to finalize our decision by December 30, "
                "so we would appreciate receiving this information at your earliest "
                "convenience.\n\n"
                "Thank you for your prompt attention to this request.\n\n"
                "Best regards"
            ),
        ),
    ),
    TestScenario(
        id=10,
        name="Internal Policy Change Announcement (Persuasive)",
        input=EmailInputData(
            intent="Announce a new remote work policy to employees",
            key_facts=[
                "New hybrid policy: 3 days in office, 2 days remote per week",
                "Effective starting March 1",
                "Flexible hours between 7 AM and 7 PM core window",
                "New home office stipend of $500 for equipment",
            ],
            tone="persuasive",
        ),
        reference=ReferenceEmail(
            subject="Exciting Update: Our New Hybrid Work Policy",
            body=(
                "Dear Team,\n\n"
                "I am thrilled to share an update that reflects our commitment to "
                "flexibility and your well-being. After listening to your feedback "
                "and analyzing productivity data, we are introducing a new hybrid "
                "work policy.\n\n"
                "Starting March 1, our new model will be 3 days in the office and "
                "2 days remote per week. This balance gives us the best of both "
                "worlds — the collaboration and energy of in-person work, plus the "
                "flexibility and focus that working from home provides.\n\n"
                "To make your remote days even more productive, we are also "
                "introducing a $500 home office stipend for equipment and setup. "
                "Additionally, we are offering flexible hours within a 7 AM to "
                "7 PM core window, so you can structure your day in a way that "
                "works best for you.\n\n"
                "We believe this policy sets us up for success while honoring the "
                "flexibility you value. More details and FAQs will follow next "
                "week.\n\n"
                "Thank you for being part of building a workplace that works for "
                "everyone.\n\n"
                "Best regards"
            ),
        ),
    ),
]


def get_all_scenarios() -> list[TestScenario]:
    """Return all 10 test scenarios.

    Returns:
        List of TestScenario objects.
    """
    return SCENARIOS
