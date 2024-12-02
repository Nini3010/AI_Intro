# Ex1: A\* Algorithm implementation

Team members: Haris Kurtagic, Nikyar Karimi, Stefan Wesely

## 1. Task Description

The file implements a sliding puzzle game solver, which can compute solutions using various heuristics like Hamming and Manhattan distances. It supports solving single puzzles and comparing heuristics' performance.

## 2. Software Architecture Diagram

The architecture primarily revolves around a class Node to represent states in the search tree and multiple utility functions to generate states, calculate costs, and evaluate solvability.

![uml_diagram](_documentation_files/uml.png "A* UML")

## 3. Module Descriptions

### Node Class:

Defines puzzle board, cost, steps, and priority for ordering.

### Functions:

#### Heuristic Functions (hamming, manhattan):

Calculates the cost to goal state using either hamming or manhattan distance.

#### PuzzleGenerators (generate_start_state, generate_goal_state,initial_state):

Produces start and goal matrices for puzzles.

#### Validators (puzzle_solvable, get_inv_count):

Checks if puzzles are solvable based on inversion count and empty tile row.

#### Solvers (solve_puzzle, solve_single_puzzle, compare_heuristics, main):

Implements A\* algorithm, considering heuristics to traverse nodes.
Holds main flow of program.

#### Utility Functions (swap, calc_cost, get_next_moves, create_node):

Aid operations like tile swapping, cost calculation, and generating successors.

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
