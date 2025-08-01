from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from agno.agent import Agent
from agno.team import Team
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.media import File
from agno.models.openai import OpenAIResponses
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.embedder.openai import OpenAIEmbedder

user_id="user_1"
session_id="session_1"

# Your Enhanced Model Definitions
class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ClauseAnalysis(BaseModel):
    clause_title: str = Field(description="Title or label of the clause")
    category: Optional[str] = Field(description="Flexible category name or keyword (free-text)")
    text: str = Field(description="Extracted clause text from the contract")
    risk_level: RiskLevel
    rationale: Optional[str] = Field(description="Explanation of why this clause is flagged")
    redline_suggestion: Optional[str] = Field(description="Suggested revision or improvement")
    playbook_match: Optional[bool] = Field(description="Does this clause align with internal guidelines?")
    requires_user_input: Optional[bool] = Field(default=False, description="Does this clause need clarification from user?")
    user_attention_score: Optional[float] = Field(ge=0.0, le=1.0, description="How much the user should pay attention to this clause (0-1)")
    agent_confidence_score: Optional[float] = Field(ge=0.0, le=1.0, description="AI's confidence in this analysis")

class RiskItem(BaseModel):
    clause_title: str
    category: Optional[str]
    risk_level: RiskLevel
    reason: Optional[str]
    user_attention_score: Optional[float]
    agent_confidence_score: Optional[float]

class ReviewSummary(BaseModel):
    overall_compliance_status: str = Field(description="Final decision: Compliant / Partial / Risky")
    executive_summary: str = Field(description="High-level summary of issues")
    total_clauses_reviewed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    clauses_needing_user_input: List[str]

class ContractReviewBaseModel(BaseModel):
    contract_type: str = Field(description="Type of the contract (e.g., NDA, SaaS)")
    jurisdiction: Optional[str] = Field(description="Applicable governing law or location")
    clause_analyses: List[ClauseAnalysis] = Field(description="Detailed analysis of clauses")
    risk_items: List[RiskItem] = Field(description="Risk-focused list of clauses")
    summary: ReviewSummary
    recommendations: List[str] = Field(description="Recommended next actions or edits")
    questions_for_user: Optional[List[str]] = Field(description="Clarification questions to refine review")

class UserContext(BaseModel):
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    domain_industry: Optional[str] = None
    nationality: Optional[str] = None
    country: Optional[str] = None
    company_size: Optional[str] = None
    year_experience: Optional[str] = None
    specific_concerns: Optional[str] = None
    additional_context: Optional[str] = None


contract_knowledge = PDFKnowledgeBase(
    path=[{
        "path": "united_educators_checklist_guide_for_reviewing_contracts.pdf",
        "metadata": {"document_type": "contract_checklist", "source": "united_educators"}
    },{
        "path": "ServicesAgreementSample.pdf",
        "metadata": {"document_type": "contract_user", "source": user_id}
    }],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="contract_review_knowledge",
        embedder=OpenAIEmbedder()
    ),
)



# Contract Analysis Agent - following agent patterns from Agno_Code.txt
contract_analyzer = Agent(
    name="Contract Analysis Agent",
    role="Analyze contract clauses and identify key terms",
    model=OpenAIResponses(id="gpt-4o-mini", ),
    knowledge=contract_knowledge,
    search_knowledge=True,
    knowledge_filters={"document_type": "contract_checklist", "source": "united_educators"},
    instructions="""
You are a contract clause analysis specialist. Your single responsibility is to extract and categorize contract clauses based on the United Educators contract review checklist.

    FOCUS ONLY ON CLAUSE IDENTIFICATION AND EXTRACTION:

    1. **Core Contract Terms Analysis:**
       - Identify parties and their legal status
       - Extract promises, rights, and obligations of each party
       - Locate contract duration, renewal terms, and performance milestones
       - Find modification procedures and requirements
       - Identify termination clauses and remedies for nonperformance
       - Extract dispute resolution mechanisms

    2. **Clause Categorization:**
       - Parties and entities
       - Payment and financial obligations
       - Goods, services, facilities descriptions
       - Duration and renewal terms
       - Modification procedures
       - Termination and breach remedies
       - Dispute resolution and governing law
       - Third-party liability and indemnification
       - Insurance requirements
       - Signature authority requirements

    3. **Text Extraction:**
       - Extract exact clause text from the contract
       - Provide clear clause titles and categories
       - Note any referenced external documents or standards
       - Identify ambiguous language that needs clarification

    Focus solely on accurate clause identification and text extraction.
    Use the contract review checklist knowledge to ensure comprehensive coverage.
    """,
)

# Risk Assessment Agent - following agent patterns from Agno_Code.txt
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

contract_storage = SqliteAgentStorage(
        table_name="contract_sessions",
        db_file="tmp/contract_memory.db"
    )

# read_agent = Agent(
#     name="Read file Agent",
#     role="Read file",
#     model=OpenAIResponses(id="gpt-4o-mini", ),
#     tools=[{"type": "file_search"}, {"type": "web_search_preview"}],
#     markdown=True,
# )

user_needs_agent = Agent(
    name="User Needs Agent",
    role="Assess user contract review needs and confirm contract type",
    model=OpenAIResponses(id="gpt-4o-mini", ),
    instructions="""
You are a user interaction specialist. Your single responsibility is to identify user clarification needs and generate relevant questions.

    FOCUS ONLY ON USER INTERACTION AND CLARIFICATION:

    1. **Contract Type Verification:**
    - Confirm the identified contract type matches user expectations
    - Ask about specific contract purpose and business context
    - Verify party roles and relationship

    2. **Clarification Questions Generation:**
    - Identify clauses needing user input or interpretation
    - Ask about internal policies and risk tolerance
    - Inquire about specific business requirements or constraints
    - Request clarification on ambiguous terms or external references

    3. **User Attention Guidance:**
    - Flag clauses requiring immediate user attention
    - Identify areas where user expertise is needed
    - Highlight decisions requiring business judgment

    4. **Question Categories:**
    - Contract type and purpose confirmation
    - Risk tolerance and internal policy alignment  
    - Business context and operational constraints
    - Clarification of ambiguous or missing terms
    - Preference for risk allocation and insurance levels

    DO NOT analyze clauses, assess risks, or make recommendations.
    Focus solely on identifying what information is needed from the user.
    Generate clear, specific questions that help refine the contract review.
    """,
    # user_message=user_context.model_dump(),
    )

# Contract Review Team - following Team patterns from Agno_Code.txt
contract_review_team = Team(
    name="Contract Review Team",
    mode="coordinate",
    # tools=[{"type": "file_search"}, {"type": "web_search_preview"}],
    model=OpenAIResponses(id="gpt-4o-mini", ),
    members=[contract_analyzer, risk_assessor,user_needs_agent],
    response_model=ContractReviewBaseModel,
    # user_message=user_context.model_dump(),
    instructions="""
    You are a contract review team that provides comprehensive legal document analysis.

    You have access to your knowledge base which contains: The user's contract document that needs to be reviewed

    IMPORTANT: Use your knowledge base to access the user's contract document. Do not use external file tools - the contract is already loaded in your knowledge base.

    Process:
    1. Access the user's contract from your knowledge base (source: user_1)
    2. Then have the Contract Analysis Agent review the document structure and detailed clause analysis
    2. Then, have the Risk Assessment Agent evaluate risks with scoring and compliance assessment
    3. Have the User Needs Agent identify areas needing user clarification and generate questions
    4. Combine findings into a structured report with:
    - Detailed clause analyses with confidence scores
    - Risk items with attention scores
    - Comprehensive summary with compliance status
    - Prioritized recommendations
    - Questions for user clarification
    5. Ensure all risk levels are properly categorized as HIGH, MEDIUM, LOW, or UNKNOWN
    6. Calculate accurate counts for the summary section
    """,
    markdown=True,
    show_members_responses=True,
    storage=contract_storage,
    debug_mode=True,
    # add_history_to_messages=True,
    knowledge=contract_knowledge,
    knowledge_filters={"document_type": "contract_user", "source": user_id}
)

contract_knowledge.load(recreate=True)

# import os

# # Get absolute path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# contract_file = os.path.join(current_dir, "ServicesAgreementSample.pdf")

contract_review_team.print_response(
    "Please analyze this contract, it a format pdf",
)