from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image

# Step 1:  Define state
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float

# Step 2: Define the graph
graph = StateGraph(BMIState)

# Step 3: Add nodes to your graph
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']
    bmi = weight/(height**2)
    state['bmi'] = round(bmi, 2)
    return state

graph.add_node('calculate_bmi', calculate_bmi) # (node, function)


# Step 4: Add edges to your graph
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', END)

# Step 5: Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state = {'weight_kg':80, 'height_m': 1.52}
final_state = workflow.invoke(initial_state)

print(final_state)

# see the graph from langgraph
Image(workflow.get_graph().draw_mermaid_png())
