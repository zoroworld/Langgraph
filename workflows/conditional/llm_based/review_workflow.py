from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv()

# --- Model ---
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- Schemas ---
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

# --- Parsers ---
parser1 = PydanticOutputParser(pydantic_object=SentimentSchema)
parser2 = PydanticOutputParser(pydantic_object=DiagnosisSchema)

# --- Prompts ---
sentiment_prompt = ChatPromptTemplate.from_template(
    "What is the sentiment of the following review?\n\nReview: {review}\n\n{format_instructions}"
)

diagnosis_prompt = ChatPromptTemplate.from_template(
    "Diagnose this negative review:\n\nReview: {review}\n\n"
    "Return issue_type, tone, and urgency.\n\n{format_instructions}"
)

# --- Chains ---
sentiment_chain = sentiment_prompt | model | parser1
diagnosis_chain = diagnosis_prompt | model | parser2

# --- State ---
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# --- Nodes ---
def find_sentiment(state: ReviewState):
    result = sentiment_chain.invoke({
        "review": f'For the following review find out the sentiment \n {state["review"]}',
        "format_instructions": parser1.get_format_instructions()
    })
    return {"sentiment": result.sentiment}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    return "positive_response" if state["sentiment"] == "positive" else "run_diagnosis"

def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""
    response = model.invoke(prompt).content
    return {"response": response}

def run_diagnosis(state: ReviewState):
    result = diagnosis_chain.invoke({
        "review": f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
""",
        "format_instructions": parser2.get_format_instructions()
    })
    return {"diagnosis": result.model_dump()}

def negative_response(state: ReviewState):
    diagnosis  = state["diagnosis"]
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt).content
    return {"response": response}

# --- Graph ---
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# --- Run Workflow ---
initial_state = {
    "review": "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
final_state = workflow.invoke(initial_state)
print(final_state)
