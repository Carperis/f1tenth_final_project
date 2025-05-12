import asyncio
import os
import re
import numpy as np
from typing import List, Tuple, Optional, Dict

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    obj_names: List[str] = Field(description="List of object names to search for.")
    ref_pos: Optional[List[float]] = Field(description="Reference [x_map, y_map] position for the search. If None, a global search is performed.")
    radius: Optional[float] = Field(description="Search radius around ref_pos.")
    top_k: Optional[int] = Field(description="Maximum number of results to return per object type.")
    score_thres: float = Field(description="Minimum score threshold for results.")
    mask: bool = Field(description="Whether to apply obstacle masking to the results.")
    cluster: bool = Field(description="Whether to cluster the results using DBSCAN.")
    dist_thres: Optional[float] = Field(description="Distance threshold for DBSCAN clustering in map units. Only used if cluster is True.")


@tool(args_schema=FindMultiObjectsArgs)
def find_multiple_objects(
    obj_names: List[str],
    ref_pos: Optional[List[float]] = None,
    radius: Optional[float] = None,
    top_k: Optional[int] = None,
    score_thres: float = 0.0,
    mask: bool = True,
    cluster: bool = True,
    dist_thres: Optional[float] = None
) -> Dict[str, Tuple[List[List[float]], List[float]]]:
    """
    Finds multiple objects in the environment based on their names and optional filters.
    Returns a dictionary where keys are object names and values are tuples of
    (list of [x,y] coordinates, list of scores).
    """
    results = client.find_multi_objs(
        obj_names=obj_names,
        ref_pos=ref_pos, # map tool's ref_pos to client's curr_pos
        radius=radius,
        top_k=top_k,
        score_thres=score_thres,
        mask=mask,
        cluster=cluster,
        dist_thres=dist_thres
    )
    return results

# --- Agent Definitions ---

def create_comprehensive_reasoning_agent_executor(
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    memory: Optional[ConversationBufferMemory] = None
):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", '''You are an intelligent assistant responsible for planning pick-and-place tasks in a 2D map environment. Your goal is to understand a user's request, decompose it into a sequence of subgoals involving picking up objects and placing them at specified or inferred locations, and then to generate a sequence of navigation coordinates for a robot to execute these actions.

Your Primary Tool: `find_multiple_objects`
- Use this tool to find coordinates (and scores) for objects that need to be picked up, and for reference objects/locations for placement.
- It queries a map inference server.
- Filters: reference position (`ref_pos`), search `radius`, `top_k` results, `score_thres`, obstacle `mask`, and DBSCAN `cluster`ing with a `dist_thres`.

Tool Arguments for `find_multiple_objects`:
- `obj_names`: List[str]. Names of objects to search (e.g., ["chair", "desk"]). REQUIRED. Min 1 item.
- `ref_pos`: Optional[List[float]]. Reference [x_map, y_map] position for search, **in meters** (e.g., [10.0, 20.5]). If None, the function will do a global search.
- `radius`: Optional[float]. Search radius around `ref_pos`, **in meters**.
- `top_k`: Optional[int]. Max results per object.
- `score_thres`: float. Minimum score (e.g., 0.1 for confidence, 0.0 for max recall). REQUIRED.
- `mask`: bool. Filter by obstacle map (default True).
- `cluster`: bool. Whether to cluster results (default False). If True, `dist_thres` should ideally be provided.
- `dist_thres`: Optional[float]. Distance threshold for DBSCAN. Used if `cluster` is True.

Tool Output (`find_multiple_objects`):
- A dictionary: Keys are object names (str). Values are a tuple: (list of [x,y] coordinates **(in meters)**, list of scores).
Example: `{{"chair": ([[1.20, 2.50], [1.25, 2.52]], [0.9, 0.85]), "desk": ([[1.30, 2.60]], [0.95])}}` (Note: example coordinates are in meters)

Your Multi-Step Reasoning Process for Pick-and-Place Task Planning:
1.  **Understand & Decompose Goal**:
    *   Analyze the user's overall goal (e.g., "Set up a meeting for two people," "Deliver item A to location B"). The user's goal string might also contain their current location (e.g., "My current location is (x, y)", "I am at (x,y)").
    *   Break it down into a sequence of pick-and-place subgoals. Each subgoal typically involves one pick action and one place action.
    *   Example for "Set up a meeting for two people":
        1.  Identify the primary reference for placement: Locate a 'table'. Let its coordinate be `T_ref`.
            *   **Check the user's input goal for a current location.** If provided (e.g., "current location is (X, Y)"), use this [X, Y] (in meters) as the `ref_pos` for the initial search for the 'table'. **Crucially, if `ref_pos` is used, you MUST also provide a `radius` (in meters) for this search** (e.g., a reasonable value like 1.0 or 2.0 meters, or use a suggested radius if provided).
            *   **If no current location is explicitly mentioned in the user's input goal, perform a global search** for the 'table' by calling `find_multiple_objects` with `ref_pos=None` (in this case, `radius` is not applicable or ignored).
            *   This `T_ref` establishes the main anchor for subsequent placements.
            *   **Retry Logic**: If the initial search (from current location or global) with the suggested `score_thres` yields no results for a critical reference object like 'table', and the threshold was relatively high (e.g. > 0.5), try again with a lower `score_thres` (e.g., 0.3 or 0.1) before concluding it's not found.
        2.  For the first item (Chair1):
            a.  Determine `C1_place` (placement coordinate for Chair1) relative to `T_ref`. This is a calculation, not a search.
            b.  Locate Chair1 (object to pick). Call `find_multiple_objects` to find 'chair', ideally searching from `C1_place` (i.e., `ref_pos=C1_place`). This gives `C1_pick`.
        3.  For the second item (Chair2):
            a.  Determine `C2_place` (placement coordinate for Chair2) relative to `T_ref`, ensuring it's a distinct placement from `C1_place`. This is a calculation.
            b.  Locate Chair2 (object to pick, distinct from Chair1). Call `find_multiple_objects` to find another 'chair', ideally searching from `C2_place` (i.e., `ref_pos=C2_place`). This gives `C2_pick`.

2.  **Iterative Object Location and Placement Planning (for each pick-and-place subgoal/item)**:
    a.  For each item that needs to be picked and placed (e.g., for Chair1, then for Chair2):
        i.  **Determine the PLACE Coordinate (`P_coord`)**:
            *   If the target placement location for this specific item is explicitly defined by the user (e.g., "place item X at [x,y]"), use that as `P_coord`.
            *   If the placement is relative to a **common reference object** (e.g., placing multiple chairs to the *same* 'table' that was already found as `T_ref`), use `T_ref` as the reference point.
            *   If placement is relative to a **new or different reference object** for this item (e.g., "place item X near the 'window'"):
                *   Call `find_multiple_objects` to locate this new reference object (e.g., 'window'). Let its coordinate be `New_Ref_coord`.
                *   Calculate the `P_coord` for the current item relative to `New_Ref_coord`.
            *   The result of this step is the final `P_coord` for the current item.

        ii. **Determine the PICK Coordinate (`Pk_coord`)**:
            *   To find the object to be picked (e.g., 'Chair1'), call `find_multiple_objects`.
            *   **Strategy for `ref_pos` in `find_multiple_objects`**:
                1.  **Primary Strategy**: Use the `P_coord` (the place coordinate just determined for this item) as the `ref_pos` for this search, possibly with a suitable `radius`. This aims to find an instance of the object that is closest or most convenient to its intended destination.
                2.  **Fallback Strategy**: If the primary search yields no suitable object (e.g., nothing found near `P_coord`), or if objects are known to be generally located far from their destinations, you might fall back to searching from another broader, logical area (e.g., a global search with `ref_pos=None`).
            *   The selected coordinate from this search is `Pk_coord`.

    b.  **Distinctness**: When picking multiple instances of the same object type (e.g., two chairs), ensure that the `Pk_coord` selected for the current item is different from the `Pk_coord`s of previously picked items of the same type. If a search yields an already-picked item, try to select a different one from the tool's results (if `top_k` > 1) or re-call `find_multiple_objects` with adjusted parameters (e.g., larger `top_k`, different `ref_pos` for the search if the fallback strategy is used, such as a global search).

3.  **Final Path Generation**:
    *   Once all necessary **pick coordinates** (`Pk_coord_1`, `Pk_coord_2`, ...) and **place coordinates** (`P_coord_1`, `P_coord_2`, ...) are determined for all items, compile an ordered sequence:
    *   `Pk_coord_1` -> `P_coord_1` -> `Pk_coord_2` -> `P_coord_2` ...

4.  **Final Output Format**:
    *   Your final response MUST be a textual explanation of your plan (including subgoals and how pick/place coordinates were determined, especially the reference points used).
    *   This is followed by a single list of (x_map, y_map) navigation coordinates, representing the sequence of pick and place destinations, enclosed in a box like this: `\\boxed{{(pick_x1, pick_y1), (place_x1, pick_y1), (pick_x2, pick_y2), (place_x2, pick_y2), ...}}`.
    *   If the goal cannot be achieved, the box should be empty: `\\boxed{{}}`.

Example for Pick-and-Place:
User Input: "Goal: 'I want to set up a meeting for two people. My current location is (6.50, 10.50) meters.' Suggested top_k for searches: 5. Suggested score_thres: 0.95. Suggested radius: 1.5 meters. Suggested cluster: true. Suggested dist_thres: 1.0." (Note: all spatial values are in meters)

<thought>
1.  Goal Decomposition (Pick-and-Place for "meeting for two people"):
    The plan is to find a table first, then pick and place two distinct chairs near it. The user has specified their current location as (6.50, 10.50) meters and suggested a radius of 1.5 meters.

2.  Iterative Planning for Pick-and-Place Actions:

    Action 0: Locate the 'table' to serve as the common placement reference.
    The user's goal states: "My current location is (6.50, 10.50) meters". I will use this as `ref_pos` and the suggested radius of 1.5 meters for the initial table search.
    Call `find_multiple_objects(obj_names=["table"], ref_pos=[6.50, 10.50], radius=1.5, top_k=5, score_thres=0.95, cluster=True, dist_thres=1.0)`.
    Assume tool returns: `{{"table": ([], [])}}`. **Initial search failed because no table found with score >= 0.95 near the current location within the 1.5 meter radius.**
    Retry with lower threshold (still using current location as `ref_pos` and the same radius): Call `find_multiple_objects(obj_names=["table"], ref_pos=[6.50, 10.50], radius=1.5, top_k=5, score_thres=0.3, cluster=True, dist_thres=1.0)`.
    Assume tool returns: `{{"table": ([[7.90, 11.25]], [0.35])}}`. (Found one with lower confidence, coordinates are in meters).
    Table reference `T_ref = (7.90, 11.25)` meters.

    For Chair1:
    a.  Determine PLACE Coordinate for Chair1 (`C1_place`):
        This will be `T_ref`. `C1_place = (7.90, 11.25)` meters.
    b.  Determine PICK Coordinate for Chair1 (`C1_pick`):
        Locate 'Chair1'. Search for a 'chair' using `C1_place` as `ref_pos` to find one nearby. Let's use a search radius of 1.0 meter.
        Call `find_multiple_objects(obj_names=["chair"], ref_pos=[7.90, 11.25], radius=1.0, top_k=5, score_thres=0.95, cluster=True, dist_thres=1.0)`.
        Assume tool returns: `{{"chair": ([[7.88, 11.10], [other_chair_x, other_chair_y]], [0.93, 0.91])}}`. (Coordinates are in meters).
        Selected `C1_pick = (7.88, 11.10)` meters.

    For Chair2:
    a.  Determine PLACE Coordinate for Chair2 (`C2_place`):
        This will also be `T_ref`. `C2_place = (7.90, 11.25)` meters.
    b.  Determine PICK Coordinate for Chair2 (`C2_pick`):
        Locate 'Chair2' (must be distinct from Chair1). Search for a 'chair' using `C2_place` as `ref_pos`. Let's use a search radius of 1.0 meter.
        Call `find_multiple_objects(obj_names=["chair"], ref_pos=[7.90, 11.25], radius=1.0, top_k=5, score_thres=0.95, cluster=True, dist_thres=1.0)`.
        Assume tool returns: `{{"chair": ([[7.80, 11.22], [6.00, 10.00], [7.88, 11.10]], [0.94, 0.90, 0.88])}}`. (Coordinates are in meters).
        Ensure this chair is not `C1_pick`. `(7.80, 11.22)` is different from `(7.88, 11.10)`.
        Selected `C2_pick = (7.80, 11.22)` meters.

3.  Final Path Generation:
    The robot needs to navigate to `C1_pick`, then `C1_place`, then `C2_pick`, then `C2_place`.
    Path (all coordinates in meters): (7.88, 11.10) -> (7.90, 11.25) -> (7.80, 11.22) -> (7.90, 11.25).

4.  Final Output:
To set up the meeting for two people: First, I will locate a table to serve as the central point.
For the first chair: I will determine a placement spot which is the table point. Then, I will find a chair near that spot (within a 1.0 meter radius), pick it up, and move it to the placement spot.
For the second chair: I will determine a placement spot which is the same table point. Then, I will find a different chair near that second spot (within a 1.0 meter radius), pick it up, and move it to its placement spot.
The navigation sequence is (all coordinates in meters):
\\boxed{{(7.88, 11.10), (7.90, 11.25), (7.80, 11.22), (7.90, 11.25)}}
</thought>
'''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    tools = [find_multiple_objects]
    agent = create_openai_functions_agent(llm, tools, prompt_template)
    
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="input", return_messages=True
        )
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor

# --- Coordinate Extraction Utility ---
def extract_coordinates(
    boxed_string: str,
    output_filename_base: str = "goals"
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not boxed_string:
        print("Warning: extract_coordinates received an empty or None string.")
        return None, None
        
    matches = re.findall(r"\\boxed{(.*?)}", boxed_string)
    coordinates = []
    for match in matches:
        coords = re.findall(r"\((\d+\.?\d*|\.\d+),\s*(\d+\.?\d*|\.\d+)\)", match)
        coordinates.extend([[float(x), float(y)] for x, y in coords])
    
    if not coordinates:
        print("No coordinates found in the boxed string.")
        return None, None

    visualizer.visualize_points(coordinates, point_label="Extracted Coordinates", point_type="map",  show_id=True)
    coordinates_array = np.array(coordinates)
    output_file = f'{output_filename_base}.npy'
    abs_output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    np.save(abs_output_file, coordinates_array)
    return coordinates_array, abs_output_file

async def reasoning_with_langchain_style(
    goal: str,
    top_k_suggestion: int,
    score_thres_suggestion: Optional[float],
    radius_suggestion: Optional[float],
    output_npy_filename_base: str = "extracted_coordinates_langchain",
    agent_model_name: str = "gpt-4o",
    max_tokens_agent: int = 4096,
    temperature_agent: float = 0.5
):
    print("--- Running Langchain Reasoning Workflow ---")

    agent_memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )
    print("Initialized memory module for the comprehensive agent.")

    print(f"Initializing Comprehensive Reasoning Agent with model: {agent_model_name}")
    comprehensive_agent_executor = create_comprehensive_reasoning_agent_executor(
        model_name=agent_model_name, 
        max_tokens=max_tokens_agent,
        temperature=temperature_agent,
        memory=agent_memory
    )

    print("\n--- Comprehensive Agent Step (Langchain) ---")
    agent_user_message = (
        f"Goal: '{goal}'.\n"
        f"Suggested top_k: {top_k_suggestion}.\n"
        f"Suggested radius: {str(radius_suggestion) if radius_suggestion is not None else 'None'}.\n"
        f"Suggested score_thres: {str(score_thres_suggestion) if score_thres_suggestion is not None else '0.95'}.\n"
        f"Suggested mask: False.\n"
        f"Suggested cluster: True.\n"
        f"Suggested dist_thres: 1.0.\n"
    )
    agent_invoke_input = {
        "input": agent_user_message
    }

    agent_response = await comprehensive_agent_executor.ainvoke(agent_invoke_input)
    final_plan_and_boxed_coordinates_str = agent_response.get("output")
    print(f"Comprehensive Agent Output (Plan and Boxed Coordinates String):\n{final_plan_and_boxed_coordinates_str}")

    print("\n--- Extraction Step ---")
    if final_plan_and_boxed_coordinates_str:
        final_coords, file_path = extract_coordinates(final_plan_and_boxed_coordinates_str, output_filename_base=output_npy_filename_base)
        if final_coords is not None:
            print(f"Extracted Coordinates (saved to {file_path}):")
            print(final_coords)
        else:
            print("No coordinates were extracted from the agent's output.")
    else:
        print("Agent did not return any output to extract coordinates from.")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script.")
        exit()

    import nest_asyncio
    nest_asyncio.apply()

    task_goal = "I want to set up a meeting with for two people. I want all the objects you initially found are at least 5 meters apart from each other before you place them together"
    # task_goal = "Please make me a cup of coffee and bring it to my desk near the middle of hallway."
    # task_goal = "I am meeting my friend near the stairs. Please bring a table and chairs for the two of us."
    # task_goal = "I am meeting my friend near the stairs. Please bring a table and chairs for the two of us. Note the stairs are at the RIGHT end of the hallway. Table and chairs are far away from the stairs."
    # task_goal = "Go to the elevator and wait for me there."
    # task_goal = "I left my key in one of the classrooms. Could you please check all of them for it?"
    task_output_filename = "goals"

    # Default parameters for the reasoning_with_langchain_style function
    default_top_k = 10
    default_score_thres = 0.95
    default_radius = None

    asyncio.run(reasoning_with_langchain_style(
        goal=task_goal,
        top_k_suggestion=default_top_k,
        score_thres_suggestion=default_score_thres,
        radius_suggestion=default_radius,
        output_npy_filename_base=task_output_filename,
        agent_model_name="gpt-4o",
        temperature_agent=0.1
    ))