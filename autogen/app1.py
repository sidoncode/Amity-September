import asyncio
import os
from tavily import AsyncTavilyClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

tavily = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

async def web_search(query: str) -> str:
    """Search the web for up-to-date facts about a topic and return short summaries."""
    response = await tavily.search(query=query, max_results=3)
    return "\n\n".join(
        f"{r['title']} ({r['url']})\n{r['content']}" for r in response["results"]
    )

async def main():
    model_client = AnthropicChatCompletionClient(
        model="claude-sonnet-5-5",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )

    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        tools=[web_search],
        reflect_on_tool_use=True,
        system_message=(
            "You write a short poem about the given topic. "
            "Use web_search first to gather a few real details. "
            "Revise the poem when given feedback."
        ),
    )
    editor = AssistantAgent(
        name="editor",
        model_client=model_client,
        system_message=(
            "Give brief, constructive feedback on the poem. "
            "If it is good enough, reply with exactly: PASS"
        ),
    )
    audit = AssistantAgent(
        name="audit",
        model_client=model_client,
        system_message=(
            "You are an auditor. Read the poem and the editor's feedback. "
            "Start your reply with exactly one verdict word on the first line: "
            "APPROVE if the feedback is constructive, or REJECT if it is not. "
            "Then, for each bullet point in the feedback, add one short line "
            "explaining why it is good or bad. "
            "Never use the words APPROVE or REJECT anywhere except the first line."
        ),
    )

    stop_rule = (
        TextMentionTermination("APPROVE", sources=["audit"])
        | TextMentionTermination("REJECT", sources=["audit"])
        | MaxMessageTermination(15)
    )
    team = RoundRobinGroupChat([writer, editor, audit], termination_condition=stop_rule)

    await Console(team.run_stream(task="Write 5 bullet points of mumbai - latest news"))
    await model_client.close()

asyncio.run(main())