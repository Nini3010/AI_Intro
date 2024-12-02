# Ex1: A\* Algorithm implementation

Team members: Haris Kurtagic, Nikyar Karimi, Stefan Wesely

## 1. Task Description

The file implements a sliding puzzle game solver, which can compute solutions using various heuristics like Hamming and Manhattan distances. It supports solving single puzzles and comparing heuristics' performance.

## 2. Software Architecture Diagram

Our Software Architecture includes only one class called Node. It contains various attributes to be used. Further helper functions are implemented. We split these helper functions into boxes for easier organization and understanding, these do not represent another class. Each box groups functions based on their specific roles in the program.

![uml_diagram](_documentation_files/uml.png "A* UML")

## 3. Module Descriptions

### Node Class:

Defines puzzle board, cost, steps, and priority for ordering.

### Functions:

#### Heuristic Functions (`hamming`, `manhattan`):

Provides evaluation metrics for guiding the search algorithm:

- `hamming`: Counts the number of misplaced tiles compared to the goal state.
- `manhattan`: Calculates the sum of distances for each tile from its goal position, offering a more accurate cost for puzzle navigation.

#### PuzzleGenerators (`generate_start_state`, `generate_goal_state`, `initial_state`):

Handles the creation of puzzle states:

- `generate_start_state`: Produces a randomized, shuffled puzzle matrix.
- `generate_goal_state`: Creates the goal configuration where tiles are ordered sequentially.
- `initial_state`: Combines start and goal state generation to initialize the puzzle setup.

#### Validators (`puzzle_solvable`, `get_inv_count`):

Checks the validity of puzzle configurations:

- `get_inv_count`: Calculates the number of inversions in a puzzle, a key metric for determining solvability.
- `puzzle_solvable`: Evaluates whether the puzzle configuration can lead to the goal state based on its inversion count and the position of the blank tile.

#### Solvers (`solve_puzzle`, `solve_single_puzzle`, `compare_heuristics`, `main`):

Implements the main logic for solving puzzles:

- `solve_puzzle`: Uses the A\* algorithm to compute the solution path, guided by a specified heuristic.
- `solve_single_puzzle`: Solves a single instance of a puzzle, displaying the steps or explaining unsolvability.
- `compare_heuristics`: Benchmarks and compares the performance of multiple heuristics over a set of random puzzles.
- `main`: Acts as the program's entry point, allowing the user to select a heuristic or perform heuristic comparisons.

#### Utility Functions (`swap`, `calc_cost`, `get_next_moves`, `create_node`):

Provides supportive functionalities for core operations:

`swap`: Swaps two tiles in a matrix to create a new configuration.
`calc_cost`: Computes the heuristic cost for a given puzzle state.
`get_next_moves`: Generates all possible successor states from the current configuration.
`create_node`: Creates a new node for a specific successor state, associating it with the parent node and heuristic cost.

## 4. Design Decisions

- Matrix representation for puzzle board state: Easily
- Class Representation for Node:
  Is used for keeping track of the path taken to the goal.
  We preferred this over a list.
- Global Variables for Goal State: Easily accessible and unchangig goal matrix and cached goal states
- Split between single puzzle and measurement of all heuristics for multiple puzzles:

## 5. Discussions

- Scalability: The current design scales with puzzle size, but performance may degrade for larger matrices due to computational and memory overhead.
- Heuristic Trade-offs: While Manhattan generally performs better than Hamming for sliding puzzles, choosing heuristics based on problem constraints could improve efficiency.
- Modularity: Functions and classes are designed modularly, making future enhancements (e.g., additional heuristics) straightforward.
- Performance Optimization: Storing visited states and open nodes in bytes minimizes memory usage.
