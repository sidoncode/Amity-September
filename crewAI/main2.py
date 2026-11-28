from crewai import Agent, Task, Crew
from crewai.flow import Flow,listen,start
MODEL = "openai/gpt-4o-mini"

class Helloflow(Flow):
    @start()
    def pick_topic(self):
        self.state["topic"] = "electric cars"
        return self.state["topic"]

    @listen(pick_topic)
    def announce(self, topic):
        print(f"Topic picked: {topic}")
        return topic    

print(Helloflow().kickoff())