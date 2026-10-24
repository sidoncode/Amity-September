from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=500)
'''
# 1 simple Question / query
print(llm.invoke("What is the capital of France?"))
'''

#2 Chat message history
history = [SystemMessage(content="You are a helpful, concise assistant.")]
while (text := input("You: ").strip()) not in ("", "quit"):
    history.append(HumanMessage(content=text))
    reply = llm.invoke(history)
    history.append(AIMessage(content=reply.content))
    print("Claude:", reply.content)