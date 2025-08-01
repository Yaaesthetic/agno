
import json
from textwrap import dedent
from agno.models.openai import OpenAIChat
from agno.agent.agent import Agent
from agno.team.team import Team
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.storage.sqlite import SqliteStorage
from pydantic import BaseModel, Field
from typing import Optional

class Product (BaseModel):
    # product_name:Optional[str] = Field(None)
    # quantity:Optional[int] = Field(None)
    product_name:str = Field(...)
    quantity:int = Field(...)


team_storage = SqliteAgentStorage(
    db_file="tmp/team_storage_shopping.db",
    # db_url=,
    mode="team",
    table_name="team_sessions"
)

user_1 = "user_1"
session_1 = "session_1"
shopping_lists = []

# Method 3: Building it step by step
user_session_shopping_list = {}
user_session_shopping_list[user_1] = {}
user_session_shopping_list[user_1][session_1] = shopping_lists
user_session_shopping_list[user_1]["count"] = 0


def add_item(team: Team, item: Product) -> str:

    
    user_id=team.user_id
    session_id=team.session_id
    
    
    if team.team_session_state[user_id]:
        if team.team_session_state[user_id][session_id]:
            return f"No shopping list found for user {user_id} and session {session_id}"
    
    if item in team.team_session_state[user_id][session_id]:
        return f"item {item.model_dump} already exist in the shopping list"
    if item not in team.team_session_state[user_id][session_id]:
        team.team_session_state[user_id][session_id].append(item)
        team.team_session_state[user_id]["count"] +=1
        return f"item {item.model_dump} is added to the shopping list"
    return f"error occuped : item {item.model_dump} couldn't be added"

def remove_item(team: Team, item: Product) -> str:
    """Remove an item from the current user's shopping list."""
    current_user_id = team.user_id
    current_session_id = team.session_id

    if team.team_session_state[current_user_id]:
        if team.team_session_state[current_user_id][current_session_id]:
            return f"No shopping list found for user {current_user_id} and session {current_session_id}"

    if item not in team.team_session_state[current_user_id][current_session_id]:
        return f"Item '{item}' not found in the shopping list for user {current_user_id} and session {current_session_id}"

    team.team_session_state[current_user_id][current_session_id].remove(item)
    team.team_session_state[current_user_id]["count"] -=1
    return f"Item {item} removed from the shopping list"

def get_shopping_list(team: Team) -> str:
    current_user_id = team.user_id
    current_session_id = team.session_id

    # Transform Product objects → dicts
    serializable_list = [
        p.model_dump() if isinstance(p, Product) else p
        for p in team.team_session_state[current_user_id][current_session_id]
    ]

    return (
        f"Shopping list for user {current_user_id} and session {current_session_id}:\n"
        f"{json.dumps(serializable_list, indent=2)}"
    )

def get_count(team: Team):
    return team.team_session_state[team.user_id]["count"]


team = Team(
    user_id=user_1,
    session_id=session_1,
    name="Shopping Team Agent",
    model=OpenAIChat("gpt-4o-mini"),
    members=[],
    # role="Personal Shopping Assistant and List Manager",
    instructions=dedent("""\
        You are a helpful shopping assistant that manages shopping lists for users.
        
        Your primary responsibilities:
        - Help users add items to their shopping list with clear descriptions
        - Remove items when requested or when they're purchased
        - Provide the current shopping list when asked
        - Keep track of item counts and quantities
        - Suggest organization of items by category (produce, dairy, meat, etc.)
        - Remind users about frequently bought items they might have forgotten
        - Help estimate shopping time and budget when possible
        
        Guidelines:
        - Always confirm when items are added or removed
        - Ask for clarification if item descriptions are unclear
        - Be proactive in suggesting related items ("You added milk, do you need bread too?")
        - Keep responses friendly and conversational
        - If the list gets long, offer to organize it by store sections
        - Notify users when they have duplicate or similar items
        
        Always maintain the shopping list state and provide helpful shopping advice.
    """),
    # context={"top_hackernews_stories": get_top_hackernews_stories},
    # # We can add the entire context dictionary to the user message
    # add_context=True,
    tools=[add_item, remove_item, get_shopping_list, get_count],
    team_session_state=user_session_shopping_list,
    show_members_responses=True,
    show_tool_calls=True,
    add_member_tools_to_system_message=True,
    debug_mode=True
)

team.run("i want to add a bread into the shopping list",stream=False)
team.run("can you tell me what i have in the shopping list",stream=False)