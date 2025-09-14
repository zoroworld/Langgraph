from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import operator
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv("GEMINI_API_KEY")
)

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

    chain = prompt | model | parser
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

    chain = prompt | model | parser
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

    chain = prompt | model | parser
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
# print(result['avg_score'])
# {'essay': 'India and AI Time\n\nNow world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.\n\nIndia have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.\n\nIn farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.\n\nBut problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.\n\nOne more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.\n\nIndia must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.\n\nIf India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.\n\nSo, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".', 'language_feedback': "The language quality of this essay is exceptional. The vocabulary is sophisticated, precise, and used effectively throughout, employing terms like 'transformative era,' 'critical juncture,' 'burgeoning base,' 'intellectual capital,' 'cutting-edge,' 'riddled with challenges,' and 'exacerbate.' The sentence structures are varied and complex, contributing to a fluid and engaging reading experience without sacrificing clarity. The essay demonstrates a masterful command of grammar, spelling, and punctuation, with no noticeable errors. Coherence and cohesion are excellent, with smooth transitions between paragraphs and a logical progression of ideas that makes the arguments easy to follow. The tone is consistently objective, authoritative, and appropriate for the subject matter. The writing is concise, impactful, and every sentence contributes meaningfully to the overall message. This essay is a model of high-quality academic writing.", 'analysis_feedback': "The essay provides a well-structured and comprehensive overview of India's position in the age of AI, touching upon its strengths, opportunities, challenges, and potential solutions. The introduction effectively sets the stage, highlighting the critical juncture India faces. The essay successfully identifies key areas such as India's intellectual capital, the role of institutions and private players, and the government's 'AI for All' strategy. The applications of AI in agriculture, healthcare, education, and smart governance are relevant and illustrate the potential benefits.\n\nHowever, the depth of analysis is somewhat limited. While the essay covers a broad range of points, it often presents them at a surface level without delving into the nuances, complexities, or specific implications. For instance:\n\n1.  **Lack of Specificity and Evidence:** While strengths like a 'vast pool of skilled engineers' are mentioned, there's no deeper analysis of *how* this talent pool is specifically geared towards cutting-edge AI, or examples of groundbreaking Indian AI innovations. Similarly, for applications, it lists sectors but lacks concrete examples of successful AI implementations, pilot projects, or the scale of their impact. This absence of specific data or case studies makes the arguments feel less robust.\n2.  **Surface-Level Exploration of Challenges:** The challenges like the 'digital divide' and 'job displacement' are correctly identified but not thoroughly analyzed. For the digital divide, the essay doesn't explore the specific socio-economic factors contributing to it in India (e.g., language barriers, affordability of devices, quality of internet infrastructure in remote areas) or specific government initiatives to bridge this gap in the context of AI adoption. For job displacement, it doesn't elaborate on *which* sectors or types of jobs are most at risk, the projected scale of displacement, or detailed strategies beyond generic 'skilling and re-skilling'.\n3.  **Policy and Regulatory Framework:** The essay mentions India 'shaping its data protection laws' but doesn't delve into the specifics of the current legal landscape, the challenges in drafting such laws for a diverse country, or specific ethical dilemmas that might arise from AI use in an Indian context (e.g., algorithmic bias in a caste-based society, surveillance concerns). The recommendations for policies are quite generic ('open datasets', 'responsible innovation') without specific proposals tailored to India's unique context.\n4.  **Geopolitical and Economic Depth:** While the introduction mentions 'geopolitical future' and the conclusion 'massive economic growth', the essay doesn't really elaborate on how India's AI strategy positions it globally against other major AI players (like the US or China), or a detailed economic model of *how* AI would drive growth, beyond general statements.\n5.  **Critical Engagement:** The essay is largely descriptive and balanced, but it refrains from critically engaging with the efficacy of existing strategies or potential shortcomings in India's current AI trajectory. For example, is the 'AI for All' strategy truly inclusive in practice? Are current research efforts sufficiently funded or coordinated?\n\nIn summary, the essay serves as an excellent foundational piece, covering all necessary points for an introductory discussion. However, to achieve greater depth, it would benefit from more specific examples, deeper analytical exploration of causes and effects, a more critical perspective on existing initiatives, and a more detailed examination of the unique socio-economic and ethical challenges pertinent to India.", 'clarity_feedback': "The essay demonstrates exceptional clarity of thought, making it remarkably easy to follow the author's arguments and grasp the multifaceted landscape of AI in India. The structure is highly logical and well-organized, progressing seamlessly from an insightful introduction that frames India's critical juncture, through its inherent strengths and promising applications, to the pressing challenges and necessary strategic responses. \n\nEach paragraph is anchored by a clear topic sentence, ensuring a smooth and coherent flow of ideas. For instance, the clear transition from outlining 'India's strengths' to illustrating 'One of the most promising applications' and subsequently detailing the 'challenges' like the 'digital divide' and 'job displacement,' showcases a well-planned and deliberate progression of thought. The arguments are consistently well-articulated and effectively supported with relevant examples, such as the mention of '5 million STEM graduates annually,' specific institutions like 'IITs, IIITs, and IISc,' major IT players, and the 'National AI Strategy.' This grounding in concrete details enhances credibility and ensures that the ideas presented are not abstract but well-substantiated.\n\nThe essay maintains a balanced and nuanced perspective, acknowledging both the 'immense promise' and 'unprecedented challenges' of AI, which significantly contributes to its thoughtful and clear presentation. The language used is precise, concise, and largely devoid of unnecessary jargon, ensuring accessibility and understanding for a broad audience. The conclusion effectively summarizes the core message, reinforcing the essay's central thesis about opportunity, responsibility, and transformation.\n\nOverall, the essay excels in its ability to present complex information in a clear, structured, and compelling manner. The logical sequencing of ideas, the use of supporting examples, and the precise language all contribute to an outstanding level of clarity.", 'overall_feedback': "This essay demonstrates **exceptional writing quality and outstanding clarity of thought.** The language is sophisticated, precise, and grammatically flawless, making it a model of high-quality academic writing. Arguments are presented with remarkable clarity, flowing logically and coherently, supported by well-articulated ideas and a balanced perspective.\n\nHowever, while the essay provides a **well-structured and comprehensive overview** of India's position in the age of AI, its **depth of analysis is somewhat limited.** The discussion often remains at a surface level, lacking specific examples, concrete data, or case studies to fully substantiate its points. Key areas such as the nuances of India's talent pool, the specific socio-economic factors behind challenges like the digital divide and job displacement, the specifics of policy and regulatory frameworks (including unique ethical dilemmas), and detailed geopolitical/economic implications require deeper exploration. The essay would significantly benefit from a more analytical, evidence-based, and critically engaged approach, moving beyond descriptive accounts to delve into the complexities and specific implications pertinent to India's AI journey.",
#  'individual_scores': [6, 10, 9],
#  'avg_score': 8.333333333333334
#  }
