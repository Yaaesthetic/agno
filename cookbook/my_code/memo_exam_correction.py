import json
import asyncio
from textwrap import dedent
from uuid import uuid4
from typing import Optional, List
from datetime import datetime

from agno.models.openai import OpenAIChat
from agno.models.anthropic.claude import Claude
from agno.models.mistral.mistral import MistralChat
from agno.agent.agent import Agent
from agno.team.team import Team
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from pydantic import BaseModel, Field
from rich.pretty import pprint
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

# **Memory Database Setup - Exactly from Agno_Code.txt**
memory_db = SqliteMemoryDb(
    table_name="memory", 
    db_file="tmp/memory.db"
)

memory = Memory(
    model=OpenAIChat("gpt-4o-mini"), 
    db=memory_db
)

# **Response Models from Agno_Code.txt**
class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str

# **Financial Research Agents with Full Memory Integration**
stock_searcher = Agent(
    name="Stock Searcher",
    model=Claude(id="claude-3-5-sonnet-20241022"),
    role="Searches the web for information on a stock.",
    tools=[YFinanceTools(cache_results=True)],
    storage=SqliteAgentStorage(
        table_name="agent_sessions", 
        db_file="tmp/persistent_memory.db"
    ),
    memory=memory,
)

web_searcher = Agent(
    name="Web Searcher",
    model=Claude(id="claude-3-5-sonnet-20241022"),
    tools=[DuckDuckGoTools(cache_results=True)],
    role="Searches the web for information on a company.",
    storage=SqliteAgentStorage(
        table_name="agent_sessions", 
        db_file="tmp/persistent_memory.db"
    ),
    memory=memory,
)

news_researcher = Agent(
    name="News Researcher",
    model=OpenAIChat("gpt-4o"),
    role="Researches financial news and market sentiment.",
    tools=[DuckDuckGoTools(cache_results=True)],
    storage=SqliteAgentStorage(
        table_name="agent_sessions", 
        db_file="tmp/persistent_memory.db"
    ),
    memory=memory,
    instructions=dedent("""\
        You are a financial news researcher specializing in market sentiment analysis.
        
        Your responsibilities:
        - Search for recent financial news and market updates
        - Analyze sentiment and market impact of news events
        - Identify key market trends and catalyst events
        - Provide context on how news affects stock performance
        
        Always provide data-driven insights with proper sourcing.
    """),
)

risk_analyst = Agent(
    name="Risk Assessment Agent",
    model=OpenAIChat("gpt-4o"),
    role="Performs risk analysis and generates investment recommendations.",
    tools=[YFinanceTools(cache_results=True)],
    storage=SqliteAgentStorage(
        table_name="agent_sessions", 
        db_file="tmp/persistent_memory.db"
    ),
    memory=memory,
    instructions=dedent("""\
        You are a risk assessment specialist for investment analysis.
        
        Your core functions:
        - Evaluate investment risks using quantitative metrics
        - Calculate risk-adjusted returns and volatility measures
        - Assess correlation risks and portfolio diversification
        - Generate investment recommendations with risk ratings
        - Monitor risk factors and provide alerts
        
        Always provide risk ratings and explain your methodology.
    """),
)

# **Team with Advanced Memory and Session Management**
session_id = "financial_research_session_1"
user_id = "portfolio_manager_1"

financial_research_team = Team(
    name="Financial Research Team",
    mode="coordinate",
    model=Claude(id="claude-3-5-sonnet-20241022"),
    storage=SqliteAgentStorage(
        table_name="team_sessions", 
        db_file="tmp/persistent_memory.db"
    ),
    members=[stock_searcher, web_searcher, news_researcher, risk_analyst],
    instructions=dedent("""\
        You are the Financial Research Team coordinator managing specialized financial analysts.
        
        Team Coordination Strategy:
        - Stock Searcher: Handles real-time market data and technical analysis
        - Web Searcher: Gathers broader company information and fundamentals  
        - News Researcher: Analyzes market sentiment and news impact
        - Risk Analyst: Performs risk assessment and investment recommendations
        
        Always add ALL stock or company information you get from team members to the shared team context.
        
        Workflow Process:
        1. First search stock market data for current prices and metrics
        2. Then gather comprehensive company information
        3. Research recent news and sentiment analysis
        4. Finally, perform risk assessment and generate recommendations
        
        Maintain persistent memory of all analyses for future reference.
    """),
    response_model=StockAnalysis,
    memory=memory,
    enable_team_history=True,
    enable_user_memories=True,
    enable_agentic_context=True,
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
    num_of_interactions_from_history=5,
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_runs=3,
)

# **Session State Management from Agno_Code.txt**
def create_financial_session(user: str, session_id: Optional[str] = None):
    """Create or continue a financial research session with memory"""
    agent_storage = SqliteStorage(
        table_name="agent_memories", 
        db_file="tmp/agents.db"
    )
    
    if session_id is None:
        existing_sessions = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]
    
    if session_id is not None:
        print(f"Continuing Session {session_id}")
    else:
        print("Started Session")
    
    return session_id

# **Real Chat Implementation with Streaming from Agno_Code.txt**
def print_chat_history(session_run):
    """Print the messages in the memory - from Agno_Code.txt"""
    console = Console()
    messages = []
    
    for m in session_run.messages:
        message_dict = m.to_dict()
        messages.append(message_dict)
    
    console.print(
        Panel(
            JSON(json.dumps(messages, indent=4)),
            title=f"Chat History for session {session_run.session_id}",
            expand=True,
        )
    )

def print_team_memory(user_id, user_memories):
    """Print team memory - from Agno_Code.txt"""
    console = Console()
    
    for user_id in list(user_memories.keys()):
        console.print(
            Panel(
                JSON(json.dumps([user_memory.to_dict() for user_memory in user_memories[user_id]], indent=4)),
                title=f"Memories for user_id {user_id}",
                expand=True,
            )
        )

# **Advanced Memory Features from Agno_Code.txt**
def print_agent_memory(agent: Agent):
    """Print the current state of agent's memory systems - from Agno_Code.txt"""
    console = Console()
    messages = []
    session_id = agent.session_id
    session_run = agent.memory.runs[session_id][-1]
    
    for m in session_run.messages:
        message_dict = m.to_dict()
        messages.append(message_dict)
    
    # Print session summary
    for user_id in list(agent.memory.summaries.keys()):
        console.print(
            Panel(
                JSON(json.dumps([summary.to_dict() for summary in agent.memory.get_session_summaries(user_id=user_id)], indent=4)),
                title=f"Summary for session_id {agent.session_id}",
                expand=True,
            )
        )

# **Multi-Session Management from Agno_Code.txt**
async def run_financial_research_system():
    """Main execution with full memory and chat functionality"""
    
    # Initialize session
    session_id = create_financial_session(user_id)
    
    # **Real-time Financial Analysis with Streaming**
    print("=== Starting Financial Research Session ===")
    
    # First analysis
    await financial_research_team.aprint_response(
        "Analyze Apple stock (AAPL) - provide comprehensive analysis including current price, company fundamentals, recent news sentiment, and investment recommendation with risk assessment.",
        stream=True,
        stream_intermediate_steps=True,
        session_id=session_id,
        user_id=user_id
    )
    
    # Print memory state
    session_run = memory.runs[session_id][-1]
    print_chat_history(session_run)
    
    # Follow-up analysis leveraging memory
    await financial_research_team.aprint_response(
        "Now compare the Apple analysis with Microsoft (MSFT) stock. Reference our previous Apple analysis and provide a comparative investment recommendation.",
        stream=True,
        stream_intermediate_steps=True,
        session_id=session_id,
        user_id=user_id
    )
    
    # Print updated memory
    session_run = memory.runs[session_id][-1]
    print_chat_history(session_run)
    
    # **Memory-based Context Continuation**
    await financial_research_team.aprint_response(
        "Based on our previous analyses, what would be the optimal portfolio allocation between Apple and Microsoft for a moderate risk investor?",
        stream=True,
        stream_intermediate_steps=True,
        session_id=session_id,
        user_id=user_id
    )
    
    # **Team Context and Member Interactions**
    print("=== Team Context ===")
    print("Team Context:", financial_research_team.memory.team_context[session_id].text)
    
    for interaction in financial_research_team.memory.team_context[session_id].member_interactions:
        print("Member Interactions:", f"{interaction.member_name}: {interaction.task} -> {interaction.response.content}")
    
    # **Session Summary Generation**
    session_summary = financial_research_team.get_session_summary(user_id=user_id, session_id=session_id)
    print("Session Summary:", session_summary.summary)
    
    # **User Memories Display**
    print_team_memory(user_id, financial_research_team.get_user_memories(user_id))

# **Interactive Chat Loop from Agno_Code.txt**
def interactive_financial_chat(user: str = "portfolio_manager"):
    """Interactive chat loop with memory display - from Agno_Code.txt"""
    
    agent_storage = SqliteStorage(
        table_name="agent_memories", 
        db_file="tmp/agents.db"
    )
    
    # Check for existing sessions
    existing_sessions = agent_storage.get_all_session_ids(user)
    if len(existing_sessions) > 0:
        session_id = existing_sessions[0]
        print(f"Continuing session: {session_id}")
    else:
        session_id = str(uuid4())
        print(f"Starting new session: {session_id}")
    
    # Enhanced agent with memory
    financial_advisor = Agent(
        user_id=user,
        session_id=session_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=dedent("""\
            You are a helpful and friendly financial advisor with excellent memory.
            - Remember important details about users and reference them naturally
            - Maintain a warm, positive tone while being precise and helpful
            - When appropriate, refer back to previous conversations and memories
            - Always be truthful about what you remember or don't remember
        """),
        tools=[YFinanceTools(cache_results=True), DuckDuckGoTools(cache_results=True)],
        storage=SqliteStorage(
            table_name="agent_memories", 
            db_file="tmp/agents.db"
        ),
        memory=Memory(
            db=SqliteMemoryDb(
                table_name="agent_memory", 
                db_file="tmp/agent_memory.db"
            ),
        ),
        enable_user_memories=True,
        enable_session_summaries=True,
        add_history_to_messages=True,
        num_history_responses=3,
        debug_mode=True,
        show_tool_calls=True,
        markdown=True,
    )
    
    print("Financial Advisor Ready! (Type 'exit' to quit)")
    print("Try: 'My portfolio includes AAPL and MSFT. What's the current status?'")
    
    exit_on = ["exit", "quit", "bye"]
    while True:
        message = input(f"\n[{user}]: ")
        if message.lower() in exit_on:
            break
            
        financial_advisor.print_response(
            message=message, 
            stream=True, 
            markdown=True
        )
        
        # Display memory state
        print_agent_memory(financial_advisor)

# **Chore Management from Shopping Team Example**
def add_chore(team: Team, chore: str, priority: str = "medium") -> str:
    """Add a chore with timestamp and priority - from Agno_Code.txt"""
    if "chores" not in team.session_state:
        team.session_state["chores"] = []  # Initialize chores list if it doesn't exist

    # Validate priority
    valid_priorities = ["low", "medium", "high"]
    if priority.lower() not in valid_priorities:
        priority = "medium"  # Default to medium if invalid

    from datetime import datetime
    chore_entry = {
        "description": chore,
        "priority": priority.lower(),
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    team.session_state["chores"].append(chore_entry)
    return f"Added chore '{chore}' with priority '{priority}'"

# **Portfolio Monitoring Tool**
def list_portfolio_items(team: Team) -> str:
    """List all items in the portfolio - adapted from shopping list example"""
    portfolio = team.team_session_state.get("portfolio", [])
    if not portfolio:
        return "The portfolio is empty."
    
    items_text = "\n".join(f"- {item}" for item in portfolio)
    return f"Current portfolio:\n{items_text}"

# **Enhanced Team with Session State Management**
portfolio_management_team = Team(
    name="Portfolio Management Team",
    mode="coordinate",
    model=OpenAIChat(id="gpt-4o-mini"),
    team_session_state={"portfolio": []},
    tools=[list_portfolio_items, add_chore],
    session_state={"chores": []},
    team_id="portfolio_team",
    members=[financial_research_team],
    show_tool_calls=True,
    markdown=True,
    instructions=dedent("""\
        You are a portfolio management team that oversees investment research and portfolio optimization.
        
        Key Responsibilities:
        - Coordinate with the Financial Research Team for analysis
        - Maintain portfolio state and track investments
        - Log all portfolio changes and research activities as chores
        - Provide comprehensive investment guidance
        
        After each completed task, use the add_chore tool to log exactly what was done with high priority.
    """),
    show_members_responses=True,
)

# **Main Execution**
if __name__ == "__main__":
    # Run comprehensive financial research system
    print("Starting Advanced Financial Research System with Memory...")
    asyncio.run(run_financial_research_system())
    
    # Optional: Start interactive chat
    # interactive_financial_chat("hedge_fund_manager")
    
    # **Memory State Inspection**
    print("\n=== Final Memory State ===")
    pprint(memory.memories)
    pprint(memory.summaries)
    pprint(memory.runs)
    
    # **Team Memory Analysis**
    print("\n=== Team Memory Analysis ===")
    pprint(financial_research_team.memory.get_messages_for_session(session_id=session_id))
    pprint(financial_research_team.memory.get_session_summaries(user_id=user_id))
    pprint(financial_research_team.memory.get_team_member_interactions_str(session_id=session_id))
    pprint(financial_research_team.memory.get_user_memories(user_id=user_id))
    pprint(financial_research_team.memory.get_runs(session_id=session_id))
