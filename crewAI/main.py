from crewai import Agent, Task, Crew

MODEL = "openai/gpt-4o-mini"

researcher = Agent(
    role="Researcher",
    goal="Find key facts about {topic}",
    backstory="You are a researcher with a keen eye for detail.",
    llm=MODEL,
)

analyst = Agent(
    role="Analyst",
    goal="Identify the main insights about {topic}",
    backstory="You are an analyst with a keen eye for detail.",
    llm=MODEL,
)

writer = Agent(
    role="Writer",
    goal="Write a clear report on {topic}",
    backstory="You are a writer with a keen eye for detail.",
    llm=MODEL,
)

research = Task(
    description="Research {topic}.",
    expected_output="5 key facts as bullet points",
    agent=researcher,
)

analysis = Task(
    description="Analyze the research on {topic}.",
    expected_output="3 main insights, including benefits and risks",
    agent=analyst,
)

report = Task(
    description="Write a report on {topic} using the research and analysis.",
    expected_output="A report of about 200 words",
    agent=writer,
    output_file="report.md",
)

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research, analysis, report],
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "the impact of AI in healthcare"})
print(result)