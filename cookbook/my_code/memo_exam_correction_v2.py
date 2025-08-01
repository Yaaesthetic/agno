import json
import asyncio
from textwrap import dedent
from agno.models.openai import OpenAIChat
from agno.agent.agent import Agent
from agno.team.team import Team
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from pydantic import BaseModel, Field
from typing import Optional, List
from agno.tools.yfinance import YFinanceTools
from rich.pretty import pprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

# Initialize Rich Console for better output formatting
console = Console()

# Session Configuration
user_1 = "user_1"
session_1 = "session_1"

# ================================
# MEMORY & STORAGE CONFIGURATION
# ================================

# Agent Memory Database - Fixed configuration
memory_db_agent = SqliteMemoryDb(
    db_file="tmp/agent_memory.db",
    table_name="agent_memory"
)

# Agent Storage Database - Fixed configuration
storage_agent = SqliteAgentStorage(
    db_file="tmp/agent_storage.db",
    table_name="agent_storage"
)

# Agent Memory System
memory_agent = Memory(
    db=memory_db_agent,
    model=OpenAIChat("gpt-4o-mini"),
)

# Team Memory Database - Fixed configuration  
memory_db_team = SqliteMemoryDb(
    db_file="tmp/team_memory.db", 
    table_name="team_memory"
)

# Team Storage Database - Fixed configuration
storage_team = SqliteAgentStorage(
    db_file="tmp/team_storage.db",
    table_name="team_storage"
)

# Team Memory System
memory_team = Memory(
    db=memory_db_team,
    model=OpenAIChat("gpt-4o-mini"),
)

# ================================
# RESPONSE MODELS
# ================================

class StockAnalysis(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    current_price: Optional[float] = Field(description="Current stock price")
    analysis: str = Field(description="Detailed financial analysis")
    recommendation: str = Field(description="Investment recommendation")
    risk_level: str = Field(description="Risk assessment: Low, Medium, High")

# ================================
# AGENT DEFINITIONS
# ================================

# Financial Analysis Agent - Simplified configuration based on Agno_Code.txt patterns
agent_finance = Agent(
    name="Financial Analysis Agent",
    role="Expert Financial Advisor and Market Analyst",
    model=OpenAIChat("gpt-4o-mini"),
    tools=[YFinanceTools(enable_all=True)],
    storage=storage_agent,
    add_memory_references=True,
    session_state=[],
    add_session_summary_references=True,
    enable_agentic_memory=True,
    enable_session_summaries=True,
    enable_user_memories=True,
    memory=memory_agent,
    instructions=dedent("""\
        You are an expert financial advisor and market analyst with access to real-time financial data.
        
        Core Responsibilities:
        - Provide comprehensive stock analysis using YFinance data
        - Analyze company fundamentals, technical indicators, and market trends
        - Offer evidence-based investment research and portfolio guidance
        - Explain complex financial concepts in accessible terms
        - Always include disclaimers about investment risks
        
        Analysis Framework:
        1. Gather current stock data (price, volume, ratios)
        2. Analyze fundamental metrics (P/E, EPS, revenue growth)
        3. Provide technical analysis insights
        4. Assess risk factors and market conditions
        5. Deliver clear, actionable recommendations
    """),
    show_tool_calls=True,
        markdown=True,
    read_chat_history=True,
)

# ================================
# TEAM CONFIGURATION - Simplified
# ================================

financial_research_team = Team(
    name="Financial Advisory Team",
    model=OpenAIChat("gpt-4o-mini"),
    members=[agent_finance],
    storage=storage_team,
    memory=memory_team,
    enable_team_history=True,
    team_session_state=[],
    read_team_history=True,
    show_members_responses=True,
    enable_agentic_memory=True,
    enable_user_memories=True,
    share_member_interactions=True,
    instructions=dedent("""\
        You are the Financial Advisory Team coordinator managing financial research.
        
        Your role:
        - Delegate financial analysis tasks to the Financial Analysis Agent
        - Provide comprehensive investment guidance
        - Ensure consistent financial advice
        - Always include appropriate investment disclaimers
        
        When users ask financial questions, forward them to the Financial Analysis Agent
        and provide a summary of their findings.
    """),
    show_tool_calls=True,
    markdown=True,
        enable_session_summaries=True,
        cache_session=True,
    debug_mode=True,
    add_history_to_messages=True,
    num_history_runs=5,
    add_member_tools_to_system_message=True,
)

# ================================
# INTERACTIVE CHAT FUNCTIONS - Fixed
# ================================

def display_welcome():
    """Display welcome message with team capabilities"""
    welcome_panel = Panel(
        dedent("""\
        🏦 **Financial Advisory Team** - Multi-Agent Research System
        
        **Available Services:**
        • Stock Analysis & Valuation
        • Portfolio Guidance & Risk Assessment
        • Technical & Fundamental Analysis
        • Investment Recommendations
        
        **Team Members:**
        🔹 Financial Analysis Agent - Stock research and market analysis
        
        **Features:**
        ✓ Persistent Memory Across Sessions
        ✓ Real-time Market Data via YFinance
        ✓ Comprehensive Risk Assessment
        
        Type 'help' for commands or ask any financial question!
        """),
        title="🚀 Welcome to Agno Financial Research System",
        border_style="green",
        padding=(1, 2)
    )
    console.print(welcome_panel)

def display_memory_stats(team_memory, agent_memory):
    """Display current memory statistics"""
    table = Table(title="📊 Memory System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Details", style="green")
    
    # Team Memory Stats
    team_runs = len(team_memory.runs.get(session_1, []))
    team_memories = len(team_memory.memories.get(user_1, []))
    
    table.add_row("Team Runs", str(team_runs), f"Session: {session_1}")
    table.add_row("Team Memories", str(team_memories), f"User: {user_1}")
    
    # Agent Memory Stats  
    agent_runs = len(agent_memory.runs.get(session_1, []))
    agent_memories = len(agent_memory.memories.get(user_1, []))
    
    table.add_row("Agent Runs", str(agent_runs), "Individual agent runs")
    table.add_row("Agent Memories", str(agent_memories), "Agent-specific memories")
    
    console.print(table)

def process_user_query(query: str, show_memory: bool = False):
    """Process user query and display results - FIXED VERSION"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description="Processing financial query...", total=None)
        
        try:
            # Execute team query
            response = financial_research_team.run(
                query, 
                session_id=session_1,
                user_id=user_1
            )
            
            progress.update(task, description="Analysis complete!", completed=100)
            
        except Exception as e:
            progress.update(task, description=f"Error: {str(e)}", completed=100)
            console.print(f"❌ Error processing query: {str(e)}")
            return
    
    # Display Response
    response_content = response.content if hasattr(response, 'content') else str(response)
    response_panel = Panel(
        response_content,
        title="💼 Financial Advisory Team Response",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(response_panel)
    
    # Display Member Responses if available - FIXED VERSION
    if hasattr(response, 'member_responses') and response.member_responses:
        console.print("\n🤝 **Team Member Contributions:**")
        
        # Handle member_responses as either list or dictionary
        if isinstance(response.member_responses, dict):
            for member, member_response in response.member_responses.items():
                content = member_response.content if hasattr(member_response, 'content') else str(member_response)
                member_panel = Panel(
                    content,
                    title=f"👤 {member}",
                    border_style="yellow",
                    padding=(0, 1)
                )
                console.print(member_panel)
        elif isinstance(response.member_responses, list):
            for i, member_response in enumerate(response.member_responses):
                content = member_response.content if hasattr(member_response, 'content') else str(member_response)
                member_panel = Panel(
                    content,
                    title=f"👤 Team Member {i+1}",
                    border_style="yellow", 
                    padding=(0, 1)
                )
                console.print(member_panel)
    
    # Display Memory Stats if requested
    if show_memory:
        console.print("\n" + "="*50)
        display_memory_stats(memory_team, memory_agent)

def display_session_history():
    """Display session conversation history"""
    try:
        messages = memory_team.get_messages_for_session(session_id=session_1)
        
        if messages:
            console.print(f"\n📚 **Session History ({len(messages)} messages):**")
            for i, message in enumerate(messages, 1):
                role = message.get('role', 'unknown')
                content = message.get('content', '')[:200] + "..." if len(message.get('content', '')) > 200 else message.get('content', '')
                console.print(f"{i}. **{role.upper()}**: {content}")
        else:
            console.print("📭 No session history available yet.")
    except Exception as e:
        console.print(f"❌ Error retrieving session history: {str(e)}")

def explore_memory_system():
    """Comprehensive memory system exploration"""
    console.print("\n🔍 **Comprehensive Memory System Analysis**")
    console.print("="*60)
    
    try:
        # Team Memory Analysis
        console.print("\n🏢 **TEAM MEMORY SYSTEM:**")
        console.print(f"📊 **Team Runs:** {len(memory_team.runs.get(session_1, []))}")
        console.print(f"🧠 **Team Memories:** {len(memory_team.memories.get(user_1, []))}")
        
        # Agent Memory Analysis
        console.print(f"\n👤 **AGENT MEMORY SYSTEM:**")
        console.print(f"📊 **Agent Runs:** {len(memory_agent.runs.get(session_1, []))}")
        console.print(f"🧠 **Agent Memories:** {len(memory_agent.memories.get(user_1, []))}")
        
    except Exception as e:
        console.print(f"❌ Error exploring memory system: {str(e)}")

# ================================
# MAIN INTERACTIVE SYSTEM - Simplified
# ================================

def interactive_chat():
    """Main interactive chat interface"""
    display_welcome()
    
    # Sample queries for demonstration
    sample_queries = [
        "Analyze Apple (AAPL) stock and provide investment recommendation",
        "What's the current price of Microsoft stock?",
        "Compare Tesla and Ford stock performance",
        "Analyze Google's financial fundamentals"
    ]
    
    console.print("\n💡 **Sample Queries to Try:**")
    for i, query in enumerate(sample_queries, 1):
        console.print(f"{i}. {query}")
    
    while True:
        console.print("\n" + "="*50)
        try:
            user_input = console.input("\n[bold green]Enter your financial query (or 'quit' to exit, 'help' for commands): [/bold green]")
        except KeyboardInterrupt:
            console.print("\n\n⚠️  System interrupted by user")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("👋 Thank you for using Agno Financial Research System!")
            break
        
        elif user_input.lower() == 'help':
            help_text = dedent("""\
            **Available Commands:**
            • 'memory' - Show memory system statistics
            • 'history' - Display session conversation history  
            • 'explore' - Comprehensive memory system analysis
            • 'demo' - Run demonstration queries
            • 'clear' - Clear console (visual only)
            • 'quit' - Exit the system
            
            **Query Examples:**
            • "Analyze [STOCK_SYMBOL] stock"
            • "What's the current price of [COMPANY_NAME]?"
            • "Compare [STOCK1] vs [STOCK2]"
            • "Analyze [COMPANY] fundamentals"
            """)
            console.print(Panel(help_text, title="📖 Help Guide", border_style="cyan"))
        
        elif user_input.lower() == 'memory':
            display_memory_stats(memory_team, memory_agent)
        
        elif user_input.lower() == 'history':
            display_session_history()
        
        elif user_input.lower() == 'explore':
            explore_memory_system()
        
        elif user_input.lower() == 'demo':
            run_demonstration()
        
        elif user_input.lower() == 'clear':
            console.clear()
            display_welcome()
        
        else:
            process_user_query(user_input, show_memory=True)

def run_demonstration():
    """Run demonstration with sample financial queries"""
    console.print("\n🎯 **Running Financial Research Demonstration**")
    
    demo_queries = [
        "Analyze Apple (AAPL) stock - provide current price and investment recommendation",
        "What are Microsoft's key financial metrics?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n📋 **Demo Query {i}:** {query}")
        console.print("-" * 60)
        process_user_query(query, show_memory=False)
        
        if i < len(demo_queries):
            time.sleep(2)  # Brief pause between queries

# ================================
# SYSTEM STARTUP AND EXECUTION
# ================================

if __name__ == "__main__":
    try:
        console.print("🚀 **Initializing Agno Financial Research System...**")
        
        # Initialize memory systems
        console.print("💾 Setting up memory and storage systems...")
        
        # Display system readiness
        console.print("✅ **System Ready!**")
        console.print(f"📍 Session ID: {session_1}")
        console.print(f"👤 User ID: {user_1}")
        
        # Start interactive chat
        interactive_chat()
        
    except KeyboardInterrupt:
        console.print("\n\n⚠️  System interrupted by user")
    except Exception as e:
        console.print(f"\n\n❌ System error: {str(e)}")
        console.print("Please check your configuration and try again.")
    finally:
        console.print("\n🔒 **System Shutdown Complete**")

# ================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================

def test_system_components():
    """Test all system components"""
    console.print("🧪 **Testing System Components...**")
    
    try:
        # Test agent memory
        test_memory = memory_agent.get_user_memories(user_id=user_1)
        console.print(f"✅ Agent Memory: {len(test_memory)} memories")
        
        # Test team memory  
        test_team_memory = memory_team.get_user_memories(user_id=user_1)
        console.print(f"✅ Team Memory: {len(test_team_memory)} memories")
        
        # Test agents
        console.print(f"✅ Financial Agent: {agent_finance.name}")
        
        # Test team
        console.print(f"✅ Financial Team: {financial_research_team.name}")
        console.print(f"✅ Team Members: {len(financial_research_team.members)}")
        
        console.print("🎉 **All Components Ready!**")
        
    except Exception as e:
        console.print(f"❌ Component test error: {str(e)}")
