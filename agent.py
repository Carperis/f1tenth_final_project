import asyncio
import os
import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Type

from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field, conlist

from map_inference_client import MapInferenceClient
from viz_map_points import PointVisualizer

# Initialize the MapInferenceClient
# Assuming the script is run from the root of the f1tenth_final_project directory
client = MapInferenceClient(
    server_url="http://127.0.0.1:1234/infer",
    obstacle_map_file="./maps/map.pgm" # Relative path to map file
)

# Initialize the PointVisualizer
visualizer = PointVisualizer(
    map_file=client.obstacle_map_file
)

# --- Langchain Tool Definitions ---

class FindMultiObjectsArgs(BaseModel):
    obj_names: List[str] = Field(title="Object Names", description="A list of object names to query.")
    curr_pos: Optional[conlist(int, min_items=2, max_items=2)] = Field(
        default=None,
        title="Current Position (X, Y)",
        description="Optional. Current position as a list of two floats [x, y] for regional search. Example: [10.0, 20.5]"
    )
    radius: Optional[float] = Field(title="Search Radius", default=None, description="Optional. Radius for the regional search around curr_pos.")
    top_k: Optional[int] = Field(title="Top K Results per Object", default=None, description="Optional. Number of top results to return for each object, sorted by score.")
    score_thres: float = Field(title="Score Threshold", default=0.0, description="Optional. Minimum score for a coordinate to be considered. Defaults to 0.0.")
    mask: bool = Field(title="Mask by Obstacle Map", default=True, description="Optional. Whether to mask results by the obstacle map. Defaults to True.")


@tool(args_schema=FindMultiObjectsArgs)
def find_multiple_objects(
    obj_names: List[str],
    curr_pos: Optional[List[float]] = None,
    radius: Optional[float] = None,
    top_k: Optional[int] = None,
    score_thres: float = 0.0,
    mask: bool = True
) -> Dict[str, Tuple[List[List[float]], List[float]]]:
    """
    Finds coordinates and scores for multiple specified object types using the map inference server.
    It queries the server for each object and applies filters like current position,
    search radius, top-k results, score threshold, and obstacle masking.
    Returns a dictionary where keys are object names and values are tuples of
    (list of [x,y] coordinates, list of scores).
    """
    print(f"Langchain Tool 'find_multiple_objects' called with: obj_names='{obj_names}', "
          f"curr_pos={curr_pos}, radius={radius}, top_k={top_k}, "
          f"score_thres={score_thres}, mask={mask}")

    # Ensure curr_pos is a list of floats if provided, matching find_multi_objs expectation
    processed_curr_pos = None
    if curr_pos is not None:
        if len(curr_pos) == 2:
            try:
                processed_curr_pos = [float(cp) for cp in curr_pos]
            except ValueError:
                print(f"Warning: curr_pos {curr_pos} could not be converted to list of floats.")
                pass
        else:
            print(f"Warning: curr_pos {curr_pos} is not a list of two elements.")

    results = client.find_multi_objs(
        obj_names=obj_names,
        curr_pos=processed_curr_pos,
        radius=radius,
        top_k=top_k,
        score_thres=score_thres,
        mask=mask
    )
    print(f"Langchain Tool 'find_multiple_objects' result: {results}")
    return results

# --- Agent Definitions ---

def create_map_observer_agent_executor(
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    temperature: float = 0.0
):
    observer_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an observer agent. Your task is to identify objects relevant to the given goal using the provided tool.

You have one tool: `find_multiple_objects`.
- Use this tool to find coordinates and scores for one or MORE types of objects.
- It queries a map inference server for each specified object.
- You can apply filters: current position (`curr_pos`), search `radius`, `top_k` results per object, a `score_thres`, and obstacle `mask`.

Tool Arguments for `find_multiple_objects`:
- `obj_names`: List[str]. Names of objects to search (e.g., ["chair", "desk"]). REQUIRED.
- `curr_pos`: Optional[List[float]]. Current [x, y] position (e.g., [10.0, 20.5]).
- `radius`: Optional[float]. Search radius around `curr_pos`.
- `top_k`: Optional[int]. Max results per object.
- `score_thres`: float. Minimum score (e.g., 0.1 for confidence, 0.0 for max recall). REQUIRED.
- `mask`: bool. Filter by obstacle map (default True).

Tool Output (`find_multiple_objects`):
The tool returns a JSON object (dictionary):
- Keys: object names (str).
- Values: A tuple: (list of [x,y] coordinates, list of scores).
Example: `{{"chair": ([[120.0, 250.0], [125.0, 252.0]], [0.9, 0.85]), "desk": ([[130.0, 260.0]], [0.95])}}`

Your Final Response Format:
Your final response MUST be a single JSON object.
- Keys: object names (str).
- Values: List of [x,y] coordinate pairs (list of floats) for that object.
- DISCARD the scores from the tool's output for your final response.

Example Thought Process and Final Output:
User Input Goal: "Find a chair and a desk."
User Input Current Location: [10.0, 20.0]
User Input Suggested top_k: 2
User Input Suggested score_thres: 0.15

1. Identify objects: "chair", "desk".
2. Plan tool call: `find_multiple_objects(obj_names=["chair", "desk"], curr_pos=[10.0, 20.0], top_k=2, score_thres=0.15)`
3. Assume tool returns: `{{ "chair": ([[120.5, 250.0], [125.0, 252.8]], [0.9, 0.85]), "desk": ([[130.0, 260.0]], [0.95]) }}`
4. Transform: Extract coordinates, discard scores.
   - "chair": `[[120.5, 250.0], [125.0, 252.8]]`
   - "desk": `[[130.0, 260.0]]`
5. Final JSON output: `{{ "chair": [[120.5, 250.0], [125.0, 252.8]], "desk": [[130.0, 260.0]] }}`
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    tools = [find_multiple_objects]
    agent = create_openai_functions_agent(llm, tools, observer_prompt_template)
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor

def create_map_planner_agent_executor(
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    temperature: float = 0.0
):
    
    # Define the prompt for the planner agent
    planner_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a task planner.
Your Planning Process:
1.  **Understand the Goal**: Determine what objects are needed, how many of each, and the overall task.
2.  **Analyze Observer Output (`coord_output`)**: Check if necessary objects are present. Deduplicate very close coordinates for the same object instance.
3.  **Select Specific Coordinates**: Based on the `goal`, choose specific object instances and their representative coordinates.
4.  **Plan Path Order**: Determine an efficient sequence for visiting selected coordinates, considering `current_location`.
5.  **Handle Insufficient Objects**: If the goal cannot be achieved, clearly state why and output an empty list of coordinates: `\\boxed{{}}`.
6.  **Format Output**: Your final output must be a single list of (x, y) pixel coordinates to visit, enclosed in a box like this: `\\boxed{{(x1.0, y1.0), (x2.5, y2.5), ...}}`.

Example 1: Meeting Setup
Goal: "Set up a meeting with a table and two chairs."
Current Location: (50.0, 50.0)
Observer Output: `{{"Table": [[791.0,1127.0], [790.5,1128.5]], "Chair": [[881.5,1053.0], [886.0,1049.5], [870.0,1078.0], [950.5,1000.0]]}}`
<think>
The goal is to set up a meeting with one table and two chairs. Current location is (50.0,50.0).
Table: select `T = (791.0,1127.0)`.
Chair: distinct candidates are `(881.5,1053.0)`, `(870.0,1078.0)`, `(950.5,1000.0)`.
Need two chairs. Choose Chair_B `(870.0,1078.0)` and Chair_C `(950.5,1000.0)`.
Path: Current (50.0,50.0) -> Chair_B (870.0,1078.0) -> Table T (791.0,1127.0) -> Chair_C (950.5,1000.0) -> Table T (791.0,1127.0).
</think>
\\boxed{{(870.0, 1078.0), (791.0, 1127.0), (950.5, 1000.0), (791.0, 1127.0)}}

Example 2: Insufficient Objects
Goal: "Find a red ball and a blue box."
Current Location: (20.0, 20.0)
Observer Output: `{{"red_ball": [[150.0,150.0]], "green_square": [[160.5,160.5]]}}`
<think>
Goal requires "red_ball" and "blue_box". "blue_box" not found. Goal cannot be achieved.
</think>
The goal "Find a red ball and a blue box." cannot be achieved as a "blue_box" was not found.
\\boxed{{}}
"""),
        MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
        ("user", "{input}"), # Single input variable for the user message
    ])

    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )
    planner_chain = LLMChain(
        llm=llm,
        prompt=planner_prompt_template,
        memory=memory,
        verbose=True
    )
    return planner_chain

# --- Coordinate Extraction Utility ---
def extract_coordinates(
    boxed_string: str,
    output_filename_base: str = "extracted_coordinates_langchain"
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not boxed_string:
        print("Warning: extract_coordinates received an empty or None string.")
        return None, None
        
    matches = re.findall(r"\\boxed{(.*?)}", boxed_string)
    coordinates = []
    for match in matches:
        # Updated regex to match floating point numbers
        coords = re.findall(r"\((\d+\.?\d*|\.\d+),\s*(\d+\.?\d*|\.\d+)\)", match)
        coordinates.extend([[float(x), float(y)] for x, y in coords]) # Convert to float
    
    if not coordinates:
        print("No coordinates found in the boxed string.")
        return None, None

    visualizer.visualize_points(coordinates, point_label="Extracted Coordinates", point_type="map",  show_id=True)
    coordinates_array = np.array(coordinates)
    output_file = f'{output_filename_base}.npy'
    abs_output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    np.save(abs_output_file, coordinates_array)
    return coordinates_array, abs_output_file

async def reasoning_with_langchain(
    goal: str,
    current_location_xy: Tuple[int, int],
    observer_top_k_suggestion: int,
    observer_score_thres_suggestion: Optional[float],
    observer_radius_suggestion: Optional[float],
    output_npy_filename_base: str = "extracted_coordinates_langchain",
    observer_model_name: str = "gpt-4o",
    planner_model_name: str = "gpt-4o",
    max_tokens_observer: int = 4096,
    max_tokens_planner: int = 4096,
    temperature_observer: float = 0.3,
    temperature_planner: float = 0.3
):
    print("--- Running Langchain Reasoning Workflow ---")

    # --- Agent Initialization ---
    print(f"Initializing Observer Agent with model: {observer_model_name}")
    observer_agent_executor = create_map_observer_agent_executor(
        model_name=observer_model_name, 
        max_tokens=max_tokens_observer,
        temperature=temperature_observer
    )
    print(f"Initializing Planner Agent with model: {planner_model_name}")
    planner_agent_executor = create_map_planner_agent_executor(
        model_name=planner_model_name,
        max_tokens=max_tokens_planner,
        temperature=temperature_planner
    )

    # --- Step 1: Observer identifies object coordinates ---
    print("\n--- Observer Step (Langchain) ---")
    observer_user_message = (
        f"Goal: '{goal}'.\n"
        f"Your current location is {current_location_xy}.\n"
        f"Suggested top_k: {observer_top_k_suggestion}.\n"
        f"Suggested radius: {str(observer_radius_suggestion) if observer_radius_suggestion is not None else 'None'}.\n"
        f"Suggested score_thres: {str(observer_score_thres_suggestion) if observer_score_thres_suggestion is not None else 'None'}."
    )
    observer_invoke_input = {
        "input": observer_user_message
    }

    observer_response = await observer_agent_executor.ainvoke(observer_invoke_input)
    object_coordinates_json_str = observer_response.get("output")
    print(f"Observer Output (JSON String): {object_coordinates_json_str}")

    # --- Step 2: Planner devises a plan and final coordinates ---
    print("\n--- Planner Step (Langchain) ---")
    if not object_coordinates_json_str:
        print("Observer did not return valid coordinates. Cannot proceed with planning.")
        return

    planner_user_message = (
        f"Your goal is: '{goal}'.\n"
        f"Your current location is: {current_location_xy}.\n"
        f"The observer found the following objects and their pixel coordinates:\n{str(object_coordinates_json_str)}"
    )
    planner_invoke_input = {
        "input": planner_user_message
    }
    planner_response_dict = await planner_agent_executor.ainvoke(planner_invoke_input)
    plan_output_str = planner_response_dict.get(planner_agent_executor.output_key)
    print(f"Planner Output (Plan and Boxed Coordinates): {plan_output_str}")

    # --- Step 3: Extract and save final coordinates ---
    print("\n--- Extraction Step ---")
    if plan_output_str:
        final_coords, file_path = extract_coordinates(plan_output_str, output_filename_base=output_npy_filename_base)
        if final_coords is not None:
            print(f"Extracted Coordinates (saved to {file_path}):")
            print(final_coords)
        else:
            print("No coordinates were extracted from the planner's output.")
    else:
        print("Planner did not return any output to extract coordinates from.")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit()

    import nest_asyncio
    nest_asyncio.apply()

    task_goal = "I want to set up a meeting with a table and two chairs. "
    task_current_location_xy = None
    task_observer_top_k = 10
    task_observer_score_thres = None
    task_observer_radius = None
    task_output_filename = "goals"

    asyncio.run(reasoning_with_langchain(
        goal=task_goal,
        current_location_xy=task_current_location_xy,
        observer_top_k_suggestion=task_observer_top_k,
        observer_score_thres_suggestion=task_observer_score_thres,
        observer_radius_suggestion=task_observer_radius,
        output_npy_filename_base=task_output_filename
    ))