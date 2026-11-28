from crewai import LLM
from crewai.flow import Flow, listen, router, start
from pydantic import BaseModel

llm = LLM(model="openai/gpt-4o-mini")


class ArticleState(BaseModel):
    topic: str = ""
    article: str = ""


class ArticleFlow(Flow[ArticleState]):
    @start()
    def write(self):
        self.state.article = llm.call(
            f"Write a clear, concise 200 word article on {self.state.topic}."
        )

    @listen(write)
    def review(self):
        self.state.article = llm.call(
            "Improve the clarity of this article. "
            f"Return only the improved article:\n{self.state.article}"
        )

    @router(review)
    def check_length(self):
        if len(self.state.article.split()) > 200:
            return "too long"
        return "ok"

    @listen("too long")
    def shorten(self):
        self.state.article = llm.call(
            f"Shorten this article to under 200 words:\n{self.state.article}"
        )
        return self.state.article

    @listen("ok")
    def finish(self):
        return self.state.article


result = ArticleFlow().kickoff(inputs={"topic": "The impact of AI in healthcare"})

print("======== RESULT ========")
print(result)