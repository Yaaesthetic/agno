from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.knowledge import KnowledgeTools
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase

from agno.vectordb.pgvector.pgvector import PgVector
from agno.vectordb.lancedb.lance_db import LanceDb
from agno.embedder.openai import OpenAIEmbedder
from agno.memory.v2.db.postgres import PostgresMemoryDb

from agno.storage.postgres import PostgresStorage
from pathlib import Path
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team.team import Team
from ..teams.memory.utils import print_chat_history, print_team_memory
from agno.run.team import TeamRunEvent, TeamRunResponse

patho=Path(__file__).read_text("guide.txt")

memory= PostgresMemoryDb(
    db_url="postgresql://postgres:postgres@localhost:5432/support_db",
    table_name="memory_agent",
)

storage= PostgresStorage(
    db_url="postgresql://postgres:postgres@localhost:5432/support_db",
    table_name="storage_agent",
    mode="team"
)

knowGuide = TextKnowledgeBase(
    vector_db=PgVector(
        db_url="postgresql://postgres:postgres@localhost:5432/support_db",
        # embedder=
        table_name="guide_data_table"
    ),
    path=patho
)

agentGuide = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    name="Guide",
    role="your are a Guider and you should to give a guide to user needs according to your knowledge",
    tools=[KnowledgeTools(
        knowledge=knowGuide,
        add_few_shot=True,
        few_shot_examples=["High: Dehydration, lung/heart disease, polycythemia vera.","Low: Anemia, blood loss, nutritional deficiency."]
    )],
    search_knowledge=True
    # update_knowledge=True
)

knowFAQ = PDFKnowledgeBase(
    vector_db=LanceDb(
        db_url="postgresql://postgres:postgres@localhost:5432/support_db",
        # embedder=
        table_name="FAQ_data_table",
        embedder=OpenAIEmbedder(id="text-embedding-3-small")
    ),
    path=patho
)
agentFAQ = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    name="product FAQ",
    role="Q & A and you should to give a guide to user needs according to your knowledge",
    tools=[KnowledgeTools(
        knowledge=knowFAQ)],
    search_knowledge=True,
    update_knowledge=True
)

agentCustomer = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    name="Customer",
    role="your are a costumer and you should to give a guide to user needs and search if needed with a soft tone",
    tools=[DuckDuckGoTools(news=False, fixed_max_results=1)],
    )

team = Team(
    # user_id=user_id,
    # session_id=session_id,
    get_member_information_tool=True,
    num_of_interactions_from_history=5,
    stream_intermediate_steps=True,
    share_member_interactions=True,

    name="Customer Support & Information Hub",
    members=[agentGuide, agentCustomer, agentFAQ],
    mode="coordinate",
    enable_agentic_context=True,
    storage=storage,
    memory=memory,

    enable_agentic_memory=True,
    enable_session_summaries=True,
    enable_team_history=True,
    enable_user_memories=True,


)

knowGuide.load(upsert=True)
knowFAQ.load(upsert=True)

def run(session_id, user_id):

    teamer: TeamRunResponse =team.run(user_id=user_id, session_id=session_id)

    print_chat_history(

        session_run=memory.run[session_id][-1]
    )

    print_team_memory(
        memories=memory.get_table(),
        user_id=user_id
    )

