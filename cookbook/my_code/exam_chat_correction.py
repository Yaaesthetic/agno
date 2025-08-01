# --- IMPORTS ---
# This section imports all the necessary classes and functions from the Agno framework and other libraries.

# Standard library imports for file paths and pretty printing.
import os
from pathlib import Path
from pprint import pprint

# Pydantic is used for creating structured data models.
# Component: Structured Output. The LLM's response will be forced into this data structure.
from pydantic import BaseModel, Field
from typing import Literal

# Core Agno components for building the system.
from agno.agent import Agent # Component: Agent. The fundamental worker unit.
from agno.team import Team   # Component: Team. The orchestrator that manages multiple agents.

# Components related to the Large Language Model.
from agno.models.openai import OpenAIChat # Component: Model & Embedder. Interfaces with OpenAI's APIs.
from agno.embedder.openai import OpenAIEmbedder

# Components for providing agents with tools.
from agno.tools.knowledge import KnowledgeTools     # Component: Tool. Allows an agent to query a knowledge base.
from agno.tools.duckduckgo import DuckDuckGoTools   # Component: Tool. Allows an agent to search the web.
from agno.tools.reasoning import ReasoningTools     # Component: Tool. Gives the agent a scratchpad to think step-by-step.

# Components for creating knowledge bases from various sources.
from agno.knowledge.url import UrlKnowledge # Component: Knowledge. Builds a knowledge base by scraping web URLs.

# Component for storing and searching text data as vectors.
from agno.vectordb.lancedb import LanceDb # Component: VectorDB. An embedded, file-based vector database.

# The high-level and low-level components for managing conversation memory.
from agno.memory.v2.memory import Memory                  # Component: Memory (High-Level). Provides intelligent memory functions like summarization.
from agno.memory.v2.db.postgres import PostgresMemoryDb   # Component: Memory (Low-Level DB Adapter). Handles the actual connection to a Postgres DB.
from agno.storage.postgres import PostgresStorage         # Component: Storage. Handles persistence of run data (logs, tool calls) to Postgres.


class EscalationTicket(BaseModel):
    """
    This is a Pydantic model. It defines a data structure for an escalation ticket.
    The `response_model` attribute on an Agent will use this to force the LLM's output into a predictable JSON format.
    The docstring itself is important, as it's passed to the LLM to explain the purpose of the data structure.
    """
    # Defines a field for the summary. The description tells the LLM what this field is for.
    summary: str = Field(description="A concise, one-sentence summary of the user's problem.")

    # Defines a field with a fixed set of possible values ("Literal"). This ensures data consistency.
    priority: Literal["Low", "Medium", "High", "Urgent"] = Field(description="The assessed priority level of the issue.")

    # Defines another string field for the recommended action.
    recommended_action: str = Field(description="The recommended next step for the human agent.")

# --- PART 1 & 2 REQUIREMENT: KNOWLEDGE BASE SETUP ---
# Here, we define the knowledge source for our FAQAgent.

# Component: Knowledge. We instantiate a knowledge base that sources its data from a web URL.
faq_knowledge = UrlKnowledge(
    # Component: VectorDB. We assign LanceDb to store the vectorized text from the URL.
    vector_db=LanceDb(
        db_url="./lancedb_support_db",  # Attribute: `db_url`. For LanceDb, this is a local file path where the database will be created.
        table_name="faq_table",        # Attribute: `table_name`. The name of the table within the database.
        # Component: Embedder. This assigns the OpenAI model that will turn text chunks into numerical vectors for searching.
        embedder=OpenAIEmbedder(id="text-embedding-3-small"), # `text-embedding-3-small` is cost-effective and powerful.
    ),
    # Attribute: `urls`. A list of web pages to scrape for information.
    urls=["https://docs.agno.ai/getting-started/what-is-agno"],
    # Optional: You could add more URLs to expand the knowledge base.
    # urls=["https://docs.agno.ai/getting-started/what-is-agno", "https://docs.agno.ai/concepts/agents"],

    # Attribute: `max_depth`. How many links deep to follow when scraping. 1 means only scrape the initial page.
    max_depth=1,
)


# --- PART 1 & 2 REQUIREMENT: PERSISTENCE SETUP (MEMORY & STORAGE) ---
# This is the corrected setup for enabling long-term memory.

# Define the connection URL for your PostgreSQL database.
DB_URL = "postgresql://postgres:postgres@localhost:5432/support_db"

# Component: Memory (Low-Level DB Adapter). This object knows how to connect to Postgres but has no conversational logic.
memory_db_adapter = PostgresMemoryDb(
    db_url=DB_URL,           # The connection string for the database.
    table_name="memory_agent_exam", # The table where conversation history will be stored.
    schema="ai"              # The database schema to use. It's good practice to isolate AI tables.
)

# Component: Memory (High-Level). THIS IS THE CRITICAL FIX.
# This `Memory` object wraps the low-level adapter and provides the intelligent methods the `Team` needs
# (like summarization, context formatting, etc.).
memory = Memory(db=memory_db_adapter) # The `db` attribute points to our low-level Postgres adapter.

# Component: Storage. This object also connects to Postgres but is used for a different purpose:
# storing detailed logs of agent/team runs (tool calls, intermediate steps, etc.).
storage = PostgresStorage(
    db_url=DB_URL,
    table_name="storage_agent_exam",
    mode="team", # Attribute: `mode`. Specifies this storage is for a team, structuring the data accordingly.
    schema="ai"
)


# --- PART 1 & 2 REQUIREMENT: AGENT DEFINITIONS ---
# Here, we define each specialized agent that will be part of our team.

# 1. FAQ Agent
faq_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"), # Component: Model. `gpt-4o-mini` is efficient for RAG tasks.
    name="FAQAgent",                   # Attribute: `name`. A unique identifier.
    role="You are a helpful assistant that answers questions based on a provided knowledge base about product features, policies, and guides.", # Attribute: `role`. The core prompt defining the agent's persona.
    tools=[KnowledgeTools(knowledge=faq_knowledge)], # Component: Tool. Gives the agent access to the `faq_knowledge` base.
    search_knowledge=True,             # Attribute: `search_knowledge`. Tells the agent to proactively search its knowledge before answering.
)

# 2. General Support Agent
general_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    name="GeneralSupportAgent",
    role="You are a general support assistant. Use your search tool to find answers to questions about recent events, news, or topics not covered in the FAQ.",
    tools=[DuckDuckGoTools(news=False, fixed_max_results=3)], # Component: Tool. Gives the agent a web search ability.
)

# 3. Escalation Agent
escalation_agent = Agent(
    model=OpenAIChat(id="gpt-4o"), # Component: Model. Using a more powerful model (`gpt-4o`) for complex reasoning and structured output.
    name="EscalationAgent",
    role="You are an escalation specialist. Your job is to analyze urgent or complex user problems and create a formal ticket. Use your reasoning tool to think first.",
    tools=[ReasoningTools()], # Component: Tool. Provides a "chain-of-thought" capability.
    response_model=EscalationTicket, # Component: Structured Output. Forces the agent's final answer to match the `EscalationTicket` Pydantic model.
)

# 4. Greeting Agent
greeting_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    name="GreetingAgent",
    role="You are a friendly conversational agent. Your role is to handle simple greetings like 'hello' or 'thank you', and respond politely if a query is completely out of scope. Be brief and friendly.",
    # This agent has no tools, as it's purely conversational.
)


# --- PART 1 & 2 REQUIREMENT: TEAM DEFINITION ---
# The team orchestrates all the defined agents.

support_team = Team(
    name="Customer Support & Information Hub",
    # Attribute: `members`. This list contains all the agent objects that belong to the team.
    members=[faq_agent, general_agent, escalation_agent, greeting_agent],
    # Attribute: `mode`. `route` is chosen for efficiency. The team will intelligently select ONE agent per query.
    mode="route",
    # Optional: Other modes have different behaviors.
    # mode="coordinate", # A manager agent would create sub-tasks for other agents. Slower but better for complex, multi-step queries.
    # mode="collaborate", # All agents would see the query and contribute. Best for brainstorming, but can be chaotic.
    instructions="You are a master router for a customer support team. Your job is to analyze the user's query and forward it to the most suitable agent. Delegate to a specialist.", # Attribute: `instructions`. The core prompt for the team's router/manager.

    # --- Attributes for Memory & Persistence ---
    memory=memory,   # Connects the team to our high-level Memory component.
    storage=storage, # Connects the team to our Storage component for run logging.

    # --- Attributes for enabling advanced features. These now work correctly because we are using the high-level `Memory` class. ---
    enable_agentic_context=True,  # Allows the team to build a dynamic context string from past interactions.
    enable_session_summaries=True,# Automatically creates summaries of long conversations to maintain context efficiently.
    enable_team_history=True,     # Persists the full conversation history to the database.
    enable_user_memories=True,    # Allows the team to create and recall specific facts about the user.

    # --- Attributes for Debugging and Transparency ---
    stream_intermediate_steps=True, # Prints the team's internal thought process in real-time.
    show_tool_calls=True,           # Explicitly shows when an agent calls a tool.
    show_members_responses=True,    # Shows the response from the selected member agent.
)


# --- PART 3: MAIN EXECUTION BLOCK ---
# This function runs the test cases to validate the system.

def main():
    """Main function to run the exam's testing part."""
    print("--- Loading Knowledge Base ---")
    # This action populates the LanceDB vector store. `upsert=True` adds new data or updates existing data if the source has changed.
    faq_knowledge.load(upsert=True)
    print("Knowledge Base loaded successfully.\n")

    print("--- Running Test Cases ---")
    
    # In a real app, this would be managed automatically. For this script, we manage it manually.
    # The `memory` component handles this under the hood when `user_id` and `session_id` are provided.
    conversation_history = []
    
    # Test Case 1: Knowledge Base Query
    print("\n--- Test Case 1: Simple FAQ Query ---")
    query1 = "What is Agno?"
    response1 = support_team.run(query1, conversation_history=conversation_history)
    pprint(response1.content) # `response.content` holds the final answer for the user.
    conversation_history.extend([("user", query1), ("assistant", response1.content)])
    
    # Test Case 2: Structured Output Query
    print("\n--- Test Case 2: Escalation for Structured Output ---")
    query2 = "I'm locked out of my account and my business is stopped because of it. This is extremely urgent, I need help now!"
    response2 = support_team.run(query2, conversation_history=conversation_history)
    pprint(response2.content) # The output will be a Pydantic `EscalationTicket` object, not a string.
    conversation_history.extend([("user", query2), ("assistant", str(response2.content))]) # Convert to string for history.
    
    # --- Part 3: Performance and Debugging ---
    print("\n--- Performance Metrics for the last run ---")
    # `response.metrics` is a dictionary containing performance data (token usage, latency) for the single `run` call.
    pprint(response2.metrics)
    print("\n--- Session Metrics (all runs combined) ---")
    # `team.session_metrics` aggregates the metrics from all `run` calls made on the `support_team` instance in this session.
    pprint(support_team.session_metrics)


# Standard Python entry point. The script starts executing here.
if __name__ == "__main__":
    # Optional: Set your OpenAI API key from an environment variable for security.
    # from dotenv import load_dotenv
    # load_dotenv()
    # assert os.getenv("OPENAI_API_KEY"), "Please set your OPENAI_API_KEY environment variable."
    
    main()

