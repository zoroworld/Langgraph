from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import operator
import os

# Step 1 :  setup of hugging face=====================================
# Load environment variables from .env
load_dotenv()
api_key = os.getenv("HF_TOKEN")
# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

# Step 2: create models=================================================

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)


llm1 = HuggingFaceEndpoint(
    repo_id="NousResearch/Hermes-4-70B",
    task="text-generation",
)

# Wrap as Chat Model
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

# Wrap as Chat Model
model2 = ChatHuggingFace(llm=llm2)


llm3 = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct-0905",
    task="text-generation",
)

# Wrap as Chat Model
model3 = ChatHuggingFace(llm=llm3)


# Step 3: Structural output (without moder openAI) =====================================================================
class EvaluationSchema(BaseModel):
    """Structured evaluation of an essay."""
    feedback: str = Field(description='Detailed feedbackfor the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)

parser = PydanticOutputParser(pydantic_object=EvaluationSchema)

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""


# Prompt
prompt = PromptTemplate(
    template=(
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\nEssay:\n{essay}\n\n{format_instructions}"
    ),
    input_variables=["essay"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

result = chain.invoke({"essay": essay})

# print(result)

# Step 3: Start of langgraph==============================================================

#state
class UPSCState(TypedDict):

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


def evaluate_language(state: UPSCState):
    prompt = PromptTemplate(
        template=(
            "Evaluate the language quality of the following essay and provide feedback "
            "and assign a score out of 10.\n\nEssay:\n{essay}\n\n{format_instructions}"
        ),
        input_variables=["essay"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model1 | parser
    output = chain.invoke({"essay": essay})

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: UPSCState):
    prompt = PromptTemplate(
        template=(
            "Evaluate the depth of analysis of the following essay and provide "
            "detailed feedback and a score out of 10.\n\nEssay:\n{essay}\n\n{format_instructions}"
        ),
        input_variables=["essay"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model2 | parser
    output = chain.invoke({"essay": essay})

    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}


def evaluate_thought(state: UPSCState):
    prompt = PromptTemplate(
        template=(
            "Evaluate the clarity of thought of the following essay and provide "
            "detailed feedback and assign a score out of 10.\n\nEssay:\n{essay}\n\n{format_instructions}"
        ),
        input_variables=["essay"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model3 | parser
    output = chain.invoke({"essay": essay})

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: UPSCState):

    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

# start the graph
graph = StateGraph(UPSCState)

#node
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()


essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India"."""


intial_state = {
    'essay': essay2
}

result = workflow.invoke(intial_state)

print(result)

