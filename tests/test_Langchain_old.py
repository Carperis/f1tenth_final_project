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


import map_inference_client as client


# --- Langchain Tool Definitions ---
# We'll define Pydantic models for the tool arguments for better validation and description.

class FindObjectCoordinatesArgs(BaseModel):
    obj_name: str = Field(title="Object Name", description="The name of the object to query.")
    current_coords_xy: Optional[conlist(int, min_items=2, max_items=2)] = Field(
        default=None, # Explicit default for Optional field
        title="Current Coordinates (X, Y)",
        description="Optional. Current coordinates as a list of two integers [x, y]."
        # min_items and max_items are now handled by conlist for validation and schema
    )
    radius: Optional[float] = Field(title="Search Radius", default=None, description="Optional. Radius for regional search around current_coords_xy.")
    top_k: Optional[int] = Field(title="Top K Results", default=None, description="Optional. Number of top results to return, sorted by score.")
    score_thres: Optional[float] = Field(title="Score Threshold", default=0.0, description="Minimum score for a coordinate to be considered. Defaults to 0.0.")

    # The validator for length check is no longer needed as conlist handles it.
    # If you had other custom validation logic for current_coords_xy, it would remain.

@tool(args_schema=FindObjectCoordinatesArgs)
def find_object_coordinates(
    obj_name: str,
    current_coords_xy: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    top_k: Optional[int] = None,
    score_thres: Optional[float] = 0.0
) -> List[Tuple[int, int]]:
    """
    Finds coordinates for a single specified object type using the map inference server.
    It queries the server for the object and applies filters like current location,
    search radius, top-k results, and score threshold.
    """
    print(f"Langchain Tool 'find_object_coordinates' called with: obj_name='{obj_name}', "
          f"current_coords_xy={current_coords_xy}, radius={radius}, "
          f"top_k={top_k}, score_thres={score_thres}")

    # Pydantic should handle type conversion for current_coords_xy if it's a list from LLM
    # but client.py expects a tuple or None.
    current_coords_xy_tuple = tuple(current_coords_xy) if current_coords_xy else None

    result_coords_xy = client.find_object_coordinates(
        obj_name=obj_name,
        current_coords_xy=current_coords_xy_tuple,
        radius=radius,
        top_k=top_k,
        score_thres=score_thres if score_thres is not None else 0.0 # Ensure float
    )
    print(f"Langchain Tool 'find_object_coordinates' result for '{obj_name}': {result_coords_xy}")
    return result_coords_xy

class SegmentAndExtractArgs(BaseModel):
    obj_names_to_process: List[str] = Field(title="Object Names to Process", description="A list of object names to segment and extract coordinates for.")
    current_coords_xy: Optional[conlist(int, min_items=2, max_items=2)] = Field(
        default=None, # Explicit default for Optional field
        title="Current Coordinates (X, Y)",
        description="Optional. Current coordinates as a list of two integers [x, y] for regional search."
        # min_items and max_items are now handled by conlist for validation and schema
    )
    radius: Optional[float] = Field(title="Search Radius", default=None, description="Optional. Radius for the regional search around current_coords_xy.")
    top_k: Optional[int] = Field(title="Top K Results per Object", default=None, description="Optional. Number of top results to return for each object, sorted by score.")
    score_thres: Optional[float] = Field(title="Score Threshold", default=0.0, description="Optional. Minimum score for a coordinate to be considered. Defaults to 0.0.")

    # The validator for length check is no longer needed as conlist handles it.


@tool(args_schema=SegmentAndExtractArgs)
def segment_and_extract_top_k_for_each_object(
    obj_names_to_process: List[str],
    current_coords_xy: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    top_k: Optional[int] = None,
    score_thres: Optional[float] = 0.0
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Segments the map based on a list of object names and then, for each object
    in that list, extracts its top-k coordinates where it was the highest-scoring
    object and met other criteria (radius, score threshold).
    """
    print(f"Langchain Tool 'segment_and_extract_top_k_for_each_object' called with: "
          f"obj_names_to_process={obj_names_to_process}, current_coords_xy={current_coords_xy}, "
          f"radius={radius}, top_k={top_k}, score_thres={score_thres}")

    current_coords_xy_tuple = tuple(current_coords_xy) if current_coords_xy else None

    all_objects_results = client.segment_and_extract_top_k_for_each_object(
        obj_names_to_process=obj_names_to_process,
        current_coords_xy=current_coords_xy_tuple,
        radius=radius,
        top_k=top_k,
        score_thres=score_thres if score_thres is not None else 0.0 # Ensure float
    )
    print(f"Langchain Tool 'segment_and_extract_top_k_for_each_object' result: {all_objects_results}")
    return all_objects_results

# --- Agent Definitions ---

def create_map_observer_agent_executor(
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    temperature: float = 0.0
):
    # Define the prompt for the observer agent
    # This uses OpenAI Functions Agent style
    observer_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an observer agent. Your task is to identify objects relevant to the given goal using the provided tools.

You have two tools to find objects:
1. `find_object_coordinates`:
   - Use this when you need to find coordinates for a SINGLE type of object.
   - If the goal requires multiple distinct object types (e.g., "a chair and a lamp"), you would call this function multiple times, once for each object type.

2. `segment_and_extract_top_k_for_each_object`:
   - Use this when the goal requires finding MULTIPLE types of objects simultaneously, especially if their relative best locations (based on competitive scoring) are important.
   - This tool first segments the map by assigning each pixel to the object from your input list that has the highest score there. Then, it extracts top-k coordinates for each object from the pixels assigned to it and meeting other criteria.

General Guidance for Tool Usage:
- Parse the goal to determine which object(s) to search for.
- Choose the most appropriate tool. If finding just one type of object (e.g., "all doors"), use `find_object_coordinates`. If finding multiple types (e.g., "a table, two chairs, and a plant"), `segment_and_extract_top_k_for_each_object` is likely more efficient and provides contextually relevant locations.
- `current_coords_xy`: If provided in the input, pass this if a localized search is beneficial.
- `top_k`: You can suggest a `top_k` value. For the multi-object tool, this applies to each object.
- `radius`: You can suggest a `radius` for a search within a certain range of `current_coords_xy`.
- `score_thres`: You MUST provide a numerical `score_thres` (e.g., 0.1 or higher for more confidence, 0.0 for maximum recall).

Output Format:
Your final response after calling the tool(s) MUST be a single JSON object where each key is an object name and its value is a list of (x, y) coordinate pairs for that object.
If you use `find_object_coordinates` multiple times, combine their results into this single JSON.
If you use `segment_and_extract_top_k_for_each_object`, its output is already in this format.

Example Thought Process and Output (using `segment_and_extract_top_k_for_each_object`):
User Input Goal: "I need to find a place to sit (a chair) and a desk for working."
User Input Current Location: (some_x, some_y)
User Input Suggested top_k: 3
User Input Suggested score_thres: 0.15

1. Identify relevant objects from Goal: "chair", "desk".
2. Choose tool: Since I need two types of objects ("chair", "desk") and their best locations considering each other, `segment_and_extract_top_k_for_each_object` is appropriate.
3. Plan function call:
   - Call `segment_and_extract_top_k_for_each_object(obj_names_to_process=["chair", "desk"], current_coords_xy=[some_x, some_y], top_k=3, score_thres=0.15)`
4. Assume the tool call returns: `{{"chair": [[120, 250], [125, 252]], "desk": [[130, 260], [131, 260]]}}`
5. This is the final JSON output to return.
"""),
        MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
        ("user", "{input}"), # Single input variable for the user message
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    tools = [find_object_coordinates, segment_and_extract_top_k_for_each_object]
    agent = create_openai_functions_agent(llm, tools, observer_prompt_template)
    # Ensure memory's input_key matches the prompt's input variable
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
6.  **Format Output**: Your final output must be a single list of (x, y) pixel coordinates to visit, enclosed in a box like this: `\\boxed{{(x1, y1), (x2, y2), ...}}`.

Example 1: Meeting Setup
Goal: "Set up a meeting with a table and two chairs."
Current Location: (50, 50)
Observer Output: `{{"Table": [[791,1127], [790,1128]], "Chair": [[881,1053], [886,1049], [870,1078], [950,1000]]}}`
<think>
The goal is to set up a meeting with one table and two chairs. Current location is (50,50).
Table: select `T = (791,1127)`.
Chair: distinct candidates are `(881,1053)`, `(870,1078)`, `(950,1000)`.
Need two chairs. Choose Chair_B `(870,1078)` and Chair_C `(950,1000)`.
Path: Current (50,50) -> Chair_B (870,1078) -> Table T (791,1127) -> Chair_C (950,1000) -> Table T (791,1127).
</think>
\\boxed{{(870, 1078), (791, 1127), (950, 1000), (791, 1127)}}

Example 2: Insufficient Objects
Goal: "Find a red ball and a blue box."
Current Location: (20, 20)
Observer Output: `{{"red_ball": [[150,150]], "green_square": [[160,160]]}}`
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
    # Planner agent typically doesn't call tools, it reasons based on input.
    # If it needed tools, they would be added here.
    # For a non-tool-using agent, we might use a simpler chain or a custom agent structure.
    # For now, let's assume it's a direct LLM call with the formatted prompt,
    # or a very simple agent that just processes the prompt.
    # Using create_openai_functions_agent without tools will make it a conversational agent.
    # Ensure memory's input_key matches the prompt's input variable
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )
    # For a no-tool planner, LLMChain is more direct
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
        
    matches = re.findall(r"\\boxed\{(.*?)\}", boxed_string)
    coordinates = []
    for match in matches:
        coords = re.findall(r"\((\d+),\s*(\d+)\)", match)
        coordinates.extend([(int(y), int(x)) for x, y in coords]) # (y,x)
    
    if not coordinates:
        print("No coordinates found in the boxed string.")
        return None, None

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
    # Format the user message for the observer
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
    object_coordinates_json_str = observer_response.get("output") # Output key might vary based on agent type
    print(f"Observer Output (JSON String): {object_coordinates_json_str}")
    # Note: The observer agent is expected to return a JSON string.
    # If it returns a dict directly from a tool, that's also fine. Planner prompt expects a string.

    # --- Step 2: Planner devises a plan and final coordinates ---
    print("\n--- Planner Step (Langchain) ---")
    if not object_coordinates_json_str:
        print("Observer did not return valid coordinates. Cannot proceed with planning.")
        return

    # Format the user message for the planner
    planner_user_message = (
        f"Your goal is: '{goal}'.\n"
        f"Your current location is: {current_location_xy}.\n"
        f"The observer found the following objects and their pixel coordinates:\n{str(object_coordinates_json_str)}"
    )
    planner_invoke_input = {
        "input": planner_user_message
    }
    planner_response_dict = await planner_agent_executor.ainvoke(planner_invoke_input) # planner_agent_executor is now an LLMChain
    plan_output_str = planner_response_dict.get(planner_agent_executor.output_key) # Default output_key for LLMChain is "text"
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
    # IMPORTANT: Set your OpenAI API key
    # Example: os.environ["OPENAI_API_KEY"] = "sk-..."

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script, e.g., by adding the line:")
        print("os.environ[\"OPENAI_API_KEY\"] = \"YOUR_ACTUAL_API_KEY\"")
        print("at the beginning of this if __name__ == \"__main__\": block (not recommended for production)")
        print("or by setting it in your system environment.")
        exit()

    # Langchain often uses asyncio, so nest_asyncio can be helpful
    import nest_asyncio
    nest_asyncio.apply()

    # --- Configuration for the reasoning task ---
    task_goal = "I want to set up a meeting with a table and two chairs."
    # task_goal = "I want to set up a party. What should I do?"
    task_current_location_xy = (512, 320) # Example (x,y)
    task_observer_top_k = 10
    task_observer_score_thres = 0.5 # Example score threshold, or None
    task_observer_radius = None # Example radius in pixels, or None
    task_output_filename = "party_setup_coordinates"

    asyncio.run(reasoning_with_langchain(
        goal=task_goal,
        current_location_xy=task_current_location_xy,
        observer_top_k_suggestion=task_observer_top_k,
        observer_score_thres_suggestion=task_observer_score_thres,
        observer_radius_suggestion=task_observer_radius,
        output_npy_filename_base=task_output_filename
        # You can also specify model names, max_tokens, and temperatures here if needed
    ))