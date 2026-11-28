from crewai import Agent,Crew,Task
from crewai_tools import TavilySearchTool
from crewai import Process


# import depenedencies

from markdown_pdf import MarkdownPdf, Section




MODEL = "openai/gpt-4o-mini"

# ranking the top5 result
search_tool = TavilySearchTool(max_results=5)

researcher = Agent(
    role="Researcher",
    goal="Find key facts about {topic}",
    backstory="You are a researcher with a keen eye for detail.",
    llm=MODEL,
    tools=[search_tool],
)

researcher_task = Task(
    description="Search about recent most policies implement By JPMC",
    expected_output="5 key facts as bullet points",
    agent=researcher,
)

AuditorAgent = Agent(
    role="Auditor",
    goal="Audit the research on {topic}.",
    backstory="You are an auditor with a keen eye for detail.",
    llm=MODEL,
)

AuditTask = Task(
    description="Audit the research on {topic} - make sure to very strict on the audit - as this is about investment banking pipeline",
    expected_output="suggest 5 most top priority bullet points reports from the Audit",
    agent=AuditorAgent,
)


crew = Crew(
    agents=[researcher, AuditorAgent],
    tasks=[researcher_task,AuditTask],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "Search about recent most policies implement By JPMC"})
print("======== RESULT ========")
print(result)

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(Section(result.raw))
pdf.save("report.pdf")
print("Saved Successfully to the PDF document")



