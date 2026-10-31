import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments


async def main():
    kernel = Kernel()
    kernel.add_service(
        AnthropicChatCompletion(
            service_id="claude",
            ai_model_id="claude-sonnet-5-5",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    )

    # Anthropic requires max_tokens on every request
    settings = AnthropicChatPromptExecutionSettings(
        service_id="claude",
        max_tokens=1024)

    result = await kernel.invoke_prompt(
        "when is the midterm elections? in USA",
        arguments=KernelArguments(settings=settings),
    )
    print(result)


asyncio.run(main())