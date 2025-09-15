from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()

# define model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#state
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = model.invoke(prompt).content

    return {'joke': response}

def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = model.invoke(prompt).content

    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id": "1"}}
final_state_1 = workflow.invoke({'topic':'pizza'}, config=config1)

# print(final_state_1)

# {
# 'topic': 'pizza',
# 'joke': 'Why did the pizza get a job?\n\nBecause it **kneaded the dough!**',
# 'explanation': 'This is a classic pun! Here\'s the breakdown:\n\n1.  **"Why did the pizza get a job?"** This sets up the question, implying the pizza needs to earn money.\n\n2.  **"Because it kneaded the dough!"** This is where the pun happens, playing on two different meanings that sound the same:\n\n    *   **"Kneaded the dough" (literal pizza meaning):** Pizza is made from dough (the mixture of flour, water, etc.). To prepare this dough, you have to *knead* it (work it with your hands). So, a pizza literally "kneads dough" as part of its existence.\n\n    *   **"Needed the dough" (slang/job meaning):** "Dough" is a common slang term for **money**. If someone "needs the dough," it means they need money, which is why they would get a job. The word "needed" (past tense of "need") sounds *exactly* like "kneaded."\n\nThe joke is funny because it uses the homophone (words that sound alike but have different meanings) "kneaded" and "needed" to create a clever switch between the literal process of making pizza and the common reason for getting a job.'
# }

print(workflow.get_state(config1))

# StateSnapshot(values=
# {'topic': 'pizza',
# 'joke': "Why did the pizza chef go broke?\n\nBecause he couldn't make enough **dough**!",
# 'explanation': 'This joke is a pun that plays on the double meaning of the word "**dough**."\n\nHere\'s the breakdown:\n\n1.  **"Dough" (Meaning 1 - Literal):** In the context of a pizza chef, "dough" refers to the mixture of flour, water, and other ingredients that is kneaded and used to make the pizza crust. A pizza chef literally makes dough all day.\n\n2.  **"Dough" (Meaning 2 - Slang):** "Dough" is also a common, informal slang term for **money**. If you don\'t have enough "dough," you don\'t have enough money.\n\nThe joke works because:\n*   **Going broke** means running out of money.\n*   The punchline, "Because he couldn\'t make enough **dough**!", sounds like he couldn\'t make enough of the *food ingredient*.\n*   But the actual reason he went broke is because he couldn\'t make enough **money** (the slang meaning of "dough").\n\nThe humor comes from the clever twist between the chef\'s literal job (making food dough) and the financial implication (not making enough money).'}, next=(),
# config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092114-3246-649e-8002-ca516f04a050'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2025-09-15T08:52:08.704304+00:00',
# parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092113-d71d-669c-8001-aa9c13b9cce8'}}, tasks=(), interrupts=())



print(list(workflow.get_state_history(config1)))
# [
#    StateSnapshot(values={'topic': 'pizza','joke': 'Why did the pizza chef break up with the pizza?\n\nBecause he said it was too **cheesy**!', 'explanation': 'This joke is a **pun**, which relies on the double meaning of the word "cheesy."\n\nHere\'s the breakdown:\n\n1.  **Literal Meaning (Pizza Context):** When you talk about a pizza being "cheesy," it literally means it has a lot of cheese. For most people, a cheesy pizza is a *good* thing!\n\n2.  **Figurative Meaning (Relationship/Idiomatic Context):** However, "cheesy" also has a common idiomatic meaning. When something (like a comment, a pick-up line, a romantic gesture, or even a joke) is described as "cheesy," it means it\'s unoriginal, overly sentimental, clichéd, or trying too hard to be charming/funny, often to the point of being a bit cringey or lame. Think of a really bad, obvious pick-up line.\n\n**The Joke\'s Humor:**\n\nThe humor comes from the unexpected switch between these two meanings. The joke personifies the pizza, treating it as if it\'s a partner in a relationship who could *say* or *do* something "cheesy" in the figurative sense.\n\nSo, the chef isn\'t complaining about too much *actual cheese* on the pizza (which would be a good thing for a pizza), but rather implying the pizza said or did something "cheesy" in the sense of being corny or unoriginal, leading to the "breakup." It\'s a play on words that makes you imagine a pizza trying to be romantic and failing spectacularly.'},next=(),config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092117-e04f-6e9e-8002-74fb3583cf2e'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2025-09-15T08:53:47.484110+00:00',parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092117-5fa5-6eb1-8001-66780cc4d70c'}},tasks=(), interrupts=()),
#    StateSnapshot(values={'topic': 'pizza', 'joke': 'Why did the pizza chef break up with the pizza?\n\nBecause he said it was too **cheesy**!'},next=('generate_explanation',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092117-5fa5-6eb1-8001-66780cc4d70c'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-09-15T08:53:33.992705+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092116-b593-639b-8000-3823628ee230'}}, tasks=(PregelTask(id='250c8aeb-eac3-297b-f628-83c28f6aa5d1', name='generate_explanation', path=('__pregel_pull', 'generate_explanation'), error=None, interrupts=(), state=None, result={'explanation': 'This joke is a **pun**, which relies on the double meaning of the word "cheesy."\n\nHere\'s the breakdown:\n\n1.  **Literal Meaning (Pizza Context):** When you talk about a pizza being "cheesy," it literally means it has a lot of cheese. For most people, a cheesy pizza is a *good* thing!\n\n2.  **Figurative Meaning (Relationship/Idiomatic Context):** However, "cheesy" also has a common idiomatic meaning. When something (like a comment, a pick-up line, a romantic gesture, or even a joke) is described as "cheesy," it means it\'s unoriginal, overly sentimental, clichéd, or trying too hard to be charming/funny, often to the point of being a bit cringey or lame. Think of a really bad, obvious pick-up line.\n\n**The Joke\'s Humor:**\n\nThe humor comes from the unexpected switch between these two meanings. The joke personifies the pizza, treating it as if it\'s a partner in a relationship who could *say* or *do* something "cheesy" in the figurative sense.\n\nSo, the chef isn\'t complaining about too much *actual cheese* on the pizza (which would be a good thing for a pizza), but rather implying the pizza said or did something "cheesy" in the sense of being corny or unoriginal, leading to the "breakup." It\'s a play on words that makes you imagine a pizza trying to be romantic and failing spectacularly.'}),), interrupts=()),
#    StateSnapshot(values={'topic': 'pizza'}, next=('generate_joke',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092116-b593-639b-8000-3823628ee230'}}, metadata={'source': 'loop', 'step': 0, 'parents': {}}, created_at='2025-09-15T08:53:16.159258+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092116-b591-628e-bfff-cb0c09552412'}}, tasks=(PregelTask(id='c938ffd1-9ad1-b75f-587b-55663b0463d4', name='generate_joke', path=('__pregel_pull', 'generate_joke'), error=None, interrupts=(), state=None, result={'joke': 'Why did the pizza chef break up with the pizza?\n\nBecause he said it was too **cheesy**!'}),), interrupts=()),
#    StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f092116-b591-628e-bfff-cb0c09552412'}}, metadata={'source': 'input', 'step': -1, 'parents': {}}, created_at='2025-09-15T08:53:16.158418+00:00', parent_config=None, tasks=(PregelTask(id='5ebabbd9-63c7-80f6-b5e5-76ba8d4a14a9', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'topic': 'pizza'}),), interrupts=())
# ]


config2 = {"configurable": {"thread_id": "2"}}
workflow.invoke({'topic':'pasta'}, config=config2)
print(workflow.get_state(config1))
print(list(workflow.get_state_history(config2)))
#[
# StateSnapshot(values={'topic': 'pasta', 'joke': 'Why did the pasta break up with the sauce?\n\nBecause it felt too *drained*!', 'explanation': 'This joke is a pun that plays on the double meaning of the word "drained."\n\nHere\'s the breakdown:\n\n1.  **Literal Meaning (for Pasta):** When you cook pasta, the very last step before adding sauce is to **drain** it of its cooking water. This is a literal, physical action you perform with pasta.\n\n2.  **Figurative Meaning (for Humans/Relationships):** In human relationships, if someone says they feel "**drained**," it means they feel emotionally exhausted, depleted of energy, or metaphorically "sapped" by the relationship. This is a common reason for people to end relationships.\n\nThe humor comes from applying this human, emotional reason for a breakup (feeling emotionally drained) to the pasta, using its literal, culinary action (being physically drained of water) as the pun. It\'s a silly, lighthearted play on words!'}, next=(), config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212b-7e7e-6cf5-8002-1f3601191229'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2025-09-15T09:02:34.098188+00:00', parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212b-3819-696c-8001-a3bff20fc109'}}, tasks=(), interrupts=()),
# StateSnapshot(values={'topic': 'pasta', 'joke': 'Why did the pasta break up with the sauce?\n\nBecause it felt too *drained*!'}, next=('generate_explanation',), config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212b-3819-696c-8001-a3bff20fc109'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-09-15T09:02:26.716669+00:00', parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212a-f5a0-6d73-8000-7387cd2cd191'}}, tasks=(PregelTask(id='8ccc21dc-d8fd-526a-b251-31d8add36706', name='generate_explanation', path=('__pregel_pull', 'generate_explanation'), error=None, interrupts=(), state=None, result={'explanation': 'This joke is a pun that plays on the double meaning of the word "drained."\n\nHere\'s the breakdown:\n\n1.  **Literal Meaning (for Pasta):** When you cook pasta, the very last step before adding sauce is to **drain** it of its cooking water. This is a literal, physical action you perform with pasta.\n\n2.  **Figurative Meaning (for Humans/Relationships):** In human relationships, if someone says they feel "**drained**," it means they feel emotionally exhausted, depleted of energy, or metaphorically "sapped" by the relationship. This is a common reason for people to end relationships.\n\nThe humor comes from applying this human, emotional reason for a breakup (feeling emotionally drained) to the pasta, using its literal, culinary action (being physically drained of water) as the pun. It\'s a silly, lighthearted play on words!'}),), interrupts=()),
# StateSnapshot(values={'topic': 'pasta'}, next=('generate_joke',), config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212a-f5a0-6d73-8000-7387cd2cd191'}}, metadata={'source': 'loop', 'step': 0, 'parents': {}}, created_at='2025-09-15T09:02:19.746638+00:00', parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212a-f59e-69df-bfff-9b64e500ae03'}}, tasks=(PregelTask(id='20e14e3d-d3bd-42b4-fba5-37fee81f510f', name='generate_joke', path=('__pregel_pull', 'generate_joke'), error=None, interrupts=(), state=None, result={'joke': 'Why did the pasta break up with the sauce?\n\nBecause it felt too *drained*!'}),), interrupts=()),
# StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1f09212a-f59e-69df-bfff-9b64e500ae03'}}, metadata={'source': 'input', 'step': -1, 'parents': {}}, created_at='2025-09-15T09:02:19.745730+00:00', parent_config=None, tasks=(PregelTask(id='d50dbc8c-7753-4fcb-9cde-a761f1c9de41', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'topic': 'pasta'}),), interrupts=())
# ]

