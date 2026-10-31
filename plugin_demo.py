import asyncio
import os
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments, kernel_function


class WeatherPlugin:
    @kernel_function(name="get_weather", description="Get the weather for a city.")
    def get_weather(self, city: Annotated[str, "City name"]) -> str:
        return f"It is 31°C and humid in {city}."  # replace with a real API call


class GreetingPlugin:
    @kernel_function(name="show_greetingMessage", description="Greeting message.")
    def show_greetingMessage(self: Annotated) -> str:
        return f"hello mate! welcome to Semantic Kernel"  # replace with a real API call



async def main():
    kernel = Kernel()
    kernel.add_service(
        AnthropicChatCompletion(
            service_id="claude",
            ai_model_id="claude-sonnet-5-5",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    )
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
    kernel.add_plugin(GreetingPlugin(),plugin_name="show_greetingMessage")

    settings = AnthropicChatPromptExecutionSettings(
        service_id="claude",
        max_tokens=1024,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),  # Claude decides when to call tools
    )

    result = await kernel.invoke_prompt(
        "greet me! with show greeting message tool",
        arguments=KernelArguments(settings=settings),
    )
    print(result)


asyncio.run(main())