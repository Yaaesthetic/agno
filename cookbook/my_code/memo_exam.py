# Problem Solving: Multi-Agent Financial Research System Design


import json
from textwrap import dedent
from agno.models.openai import OpenAIChat
from agno.agent.agent import Agent
from agno.team.team import Team
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage
from pydantic import BaseModel, Field
from typing import Optional
from agno.tools.yfinance import YFinanceTools
# from agno.memory.v2.summarizer import SessionSummarizer
# from agno.memory.v2.manager import MemoryManager
from rich.pretty import pprint

from rich.pretty import pprint
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

user_1 = "user_1"
session_1 = "session_1"


memory_db_agent=SqliteMemoryDb(
    db_file="tmp/storage_financial_agent.db",
    table_name="financial_agent",
)

storage_agent=SqliteAgentStorage(
    db_file="tmp/memory_financial_agent.db",
    mode="agent",
    table_name="financial_agent_memory",   
)

memory_agent=Memory(
    debug_mode=True,
    # memory_manager=MemoryManager(   
    # ),
    # summarizer=SessionSummarizer(
    # )
    model=OpenAIChat("gpt-4o-mini"),
    db=memory_db_agent, 
)

memory_db_team=SqliteMemoryDb(
    db_file="tmp/storage_financial_team.db",
    table_name="financial_team",
)

storage_team=SqliteAgentStorage(
    db_file="tmp/memory_financial_team",
    mode="team",
    table_name="financial_team_memory",
)

memory_team=Memory(
    debug_mode=True,
    # memory_manager=MemoryManager(   
    # ),
    # summarizer=SessionSummarizer(
    # )
    model=OpenAIChat("gpt-4o-mini"),
    db=memory_db_team,
)


agent_finance = Agent(
    storage=storage_agent,
    memory=memory_agent,
    model=OpenAIChat("gpt-4o-mini"),
    tools=[YFinanceTools(
        enable_all=True,
    )],
    add_memory_references=True,
    add_session_summary_references=True,
    enable_agentic_memory=True,
    enable_session_summaries=True,
    enable_user_memories=True,
    name="Financial Analysis Agent",
    role="Personal Finance Advisor and Market Analyst",
    instructions=dedent("""\
        You are an expert financial advisor and market analyst with access to real-time financial data.
        
        Your core responsibilities:
        - Provide comprehensive stock analysis and market insights
        - Help users understand financial metrics and ratios
        - Analyze company fundamentals, technical indicators, and market trends
        - Offer investment research and portfolio guidance
        - Explain complex financial concepts in simple terms
        - Monitor market news and economic indicators
        
        Use your YFinance tools to access real-time market data and provide data-driven insights.
    """),
    # session_state=session_agent,
    markdown=True,
    read_chat_history=True,
)

class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str

financial_research_team = Team(
    markdown=True,
    user_id=user_1,
    session_id=session_1,
    name="Financial Advisory Team",
    model=OpenAIChat("gpt-4o-mini"),
    members=[agent_finance], # I'm gonna added other agent for finance needs later
    enable_team_history=True,
    # team_session_state=[],
    read_team_history=True,
    storage=storage_team,
    memory=memory_team,
    show_members_responses=True,
    enable_agentic_memory=True,
    enable_user_memories=True,
    share_member_interactions=True,
    # get_member_information_tool=True,
    instructions=dedent("""\
        You are the Financial Advisory Team coordinator, managing a team of specialized financial agents.
        
        Team Purpose:
        - Provide comprehensive financial analysis and investment guidance
        - Coordinate between different financial specialists when more agents are added
        - Ensure consistent and coherent financial advice across all team members
        - Maintain user context and preferences across financial discussions
        
        Current Team Members:
        - Financial Analysis Agent: Handles market analysis, stock research, and investment insights
        
        Team Coordination Guidelines:
        - Always introduce the relevant team member when delegating tasks
        - Ensure all financial advice is consistent across team members
        - Synthesize insights from multiple agents when applicable
        - Maintain conversation context and user financial goals
        - Escalate complex multi-disciplinary financial questions to appropriate specialists
        
        User Interaction:
        - Greet users and understand their financial needs and goals
        - Route specific questions to the most appropriate team member
        - Provide summaries that combine insights from multiple agents
        - Remember user preferences and investment history
        - Always remind users about investment risks and the importance of professional advice
        
        Team Expansion Ready:
        - When new agents are added (portfolio management, tax planning, retirement planning, etc.)
        - Coordinate their interactions and ensure no conflicting advice
        - Maintain a unified team approach to financial guidance
        
        Start each interaction by understanding what financial assistance the user needs today.
    """),
    cache_session=True,
    mode='route', # i use route because i have one agent at this moment and because we want it to be specific
    enable_session_summaries=True,
        cache_session=True,
    debug_mode=True,
    show_tool_calls=True,
    add_history_to_messages=True,
    num_history_runs=5,
    add_member_tools_to_system_message=True,
    response_model=StockAnalysis,
    
)

# team.run("search on the finance of google",session_id=session_1,user_id=user_1)

# pprint(team.run_response.content)
# pprint("---"*10)
# pprint(team.run_response.member_responses)
# pprint("---"*5)
# pprint(team.run_response.metrics)
# pprint("---"*10)
# pprint(team.memory.memories)
# pprint("---"*5)
# pprint(team.memory.delete_memories)
# pprint("---"*10)
# pprint(team.memory.summaries)
# pprint("---"*5)

# pprint(team.memory.runs)
# pprint("---"*10)
# pprint(team.memory.runs[session_1][-1])
# pprint("---"*5)

# pprint(team.memory.runs[session_1][-1].messages)
# pprint("---"*10)
# pprint(team.memory.runs[session_1][-1].tools)
# pprint("---"*5)

# pprint(team.memory.runs[session_1][-1].member_responses)
# pprint("---"*10)
# pprint(team.memory.runs[session_1][-1].metrics)
# pprint("---"*5)

# pprint(team.memory.runs[session_1][-1].events)
# pprint("---"*10)
# pprint(team.memory.runs[session_1][-1].status)
# pprint("---"*5)

# pprint(team.memory.get_messages_for_session(session_id=session_1))
# pprint("---"*10)
# pprint(team.memory.get_session_summaries(user_id=user_1))
# pprint("---"*5)

# pprint(team.memory.get_team_member_interactions_str(session_id=session_1))
# pprint("---"*10)
# pprint(team.memory.get_user_memories(user_id=user_1))
# pprint("---"*5)
# pprint(team.memory.get_runs(session_id=session_1))

