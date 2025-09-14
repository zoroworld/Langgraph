from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
import operator, os

load_dotenv()

# --- Models ---
generated_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

optimizer_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

evaluator_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- Evaluation Schema ---
class EvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(description="Whether the tweet is good enough")
    feedback: str = Field(description="Feedback on how to improve the tweet")

structured_evaluator = evaluator_model.with_structured_output(EvaluationSchema)

# --- State ---
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

# --- Nodes ---
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]
    response = generated_model.invoke(messages).content
    return {"tweet": response, "tweet_history": [response]}

def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. Evaluate based on humor, originality, virality, and format."),
        HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

### Respond ONLY in JSON with keys:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses
""")
    ]
    response = structured_evaluator.invoke(messages)
    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback,
        "feedback_history": [response.feedback],
    }

def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]
    response = optimizer_model.invoke(messages).content
    iteration = state["iteration"] + 1
    return {"tweet": response, "iteration": iteration, "tweet_history": [response]}

def route_evaluation(state: TweetState):
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"

# --- Graph ---
graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", route_evaluation, {"approved": END, "needs_improvement": "optimize"})
graph.add_edge("optimize", "evaluate")

workflow = graph.compile()

# --- Run ---
initial_state = {
    "topic": "My wifi is slower than government paperwork",
    "iteration": 1,
    "max_iteration": 5,
    "tweet_history": [],
    "feedback_history": [],
}
final_state = workflow.invoke(initial_state)

print(final_state)

for tweet in final_state ['tweet_history']:
    print(tweet)