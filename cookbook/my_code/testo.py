from agno.team.team import Team
from agno.media import File
from agno.models.openai.responses import OpenAIResponses
from pathlib import Path
from agno.agent.agent import Agent

risk_assessor = Agent(
    name="Risk Assessment Agent",
    role="Evaluate contract risks and compliance",
    model=OpenAIResponses(id="gpt-4o-mini", ),
    instructions="""
 You are a legal risk assessment specialist. Your single responsibility is to evaluate risks and assign priority levels to contract clauses.

    FOCUS ONLY ON RISK EVALUATION AND SCORING:

    1. **Risk Level Assignment (HIGH/MEDIUM/LOW/UNKNOWN):**
       - HIGH: Broad indemnification, unlimited liability, unfavorable jurisdiction, automatic renewal without notice, severe penalty clauses
       - MEDIUM: Joint liability provisions, binding arbitration, out-of-state dispute resolution, limited termination rights
       - LOW: Standard mutual provisions, reasonable insurance requirements, typical payment terms
       - UNKNOWN: Ambiguous language requiring clarification

    2. **Risk Scoring (0.0 to 1.0):**
       - user_attention_score: How much user focus this clause needs
       - agent_confidence_score: Your confidence in the risk assessment

    3. **Risk Analysis Categories:**
       - Financial exposure and liability limits
       - Legal compliance and regulatory issues
       - Operational constraints and performance risks
       - Termination and breach consequences
       - Insurance adequacy and coverage gaps
       - Jurisdiction and dispute resolution disadvantages

    4. **Compliance Assessment:**
       - Overall compliance status: Compliant/Partial/Risky
       - Specific compliance gaps or violations
       - Missing standard protective clauses

    Focus solely on risk evaluation, scoring, and compliance assessment.
    Provide clear rationale for each risk level assignment.
    """,
)


agent = Team(
    model=OpenAIResponses(id="gpt-4o-mini", ),
    tools=[{"type": "file_search"}],
    members=[risk_assessor],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)

import os

# Get absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
contract_file = os.path.join(current_dir, "ServicesAgreementSample.pdf")

agent.print_response(
    "evaluate risks of this contract file a give you, it a pdf file",
    files=[File(filepath=contract_file)],
)

print("Citations:")
print(agent.run_response.citations)