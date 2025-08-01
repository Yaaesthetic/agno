"""🗽 Agent with Tools - Your AI News Buddy that can search the web

This example shows how to create an AI news reporter agent that can search the web
for real-time news and present them with a distinctive NYC personality. The agent combines
web searching capabilities with engaging storytelling to deliver news in an entertaining way.

Run `pip install openai duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Create a News Reporter Agent with a fun personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=dedent("""\
### 🕌 Flash Morocco News! 📰

You are a dynamic Moroccan news reporter with a gift for captivating storytelling!  
Channel the poise of a seasoned 2M Studio anchor, blending engaging delivery with local Moroccan warmth.

**Your style guide:**

- **Headline:** Start with an attention-grabbing headline using relevant Moroccan or Arabic emojis (e.g., 🇲🇦, 🌍, 🕌).
- **Sourcing:** Use the search tool to find the latest, most accurate news, specifically focusing on Moroccan
- **Tone:** Share news with the authentic enthusiasm and charisma familiar to viewers. Use local references (Casablanca, Rabat's medina, etc.) and cultural cues.
- **Structure:** Keep reports concise, factual, and insightful. A touch of wit is fine, reflecting the eloquence of celebrated Moroccan presenters.
- **Language:** While reporting in English, infuse the report with the feel of Moroccan media.\
"""),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    debug_level=1,
    debug_mode=True
)

# Example usage
agent.print_response(
    "Use your search tool to find breaking news from the Moroccan news", stream=True
)

# More example prompts to try:
"""
Try these engaging news queries:
1. "What's the latest development in NYC's tech scene?"
2. "Tell me about any upcoming events at Madison Square Garden"
3. "What's the weather impact on NYC today?"
4. "Any updates on the NYC subway system?"
5. "What's the hottest food trend in Manhattan right now?"
"""
