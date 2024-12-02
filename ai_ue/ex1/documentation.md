# Ex1: A\* Algorithm implementation

Team members: Haris Kurtagic, Nikyar Karimi, Stefan Wesely

## 1. Task Description

The file implements a sliding puzzle game solver, which can compute solutions using various heuristics like Hamming and Manhattan distances. It supports solving single puzzles and comparing heuristics' performance.

## 2. Software Architecture Diagram

The architecture primarily revolves around a class Node to represent states in the search tree and multiple utility functions to generate states, calculate costs, and evaluate solvability. Here's a logical flow:

Node Class - Represents puzzle states.
Heuristics Functions - Hamming and Manhattan distances for evaluation.
Puzzle Generation - Creates start and goal states.
Solver - Employs A\* algorithm to find solutions.
(For actual visual representation, let me know to create one!)

## 3. Module Descriptions

Node Class: Defines puzzle state, cost, steps, and ordering logic.
Heuristic Functions (hamming, manhattan): Calculates the cost to goal state using different approaches.
Puzzle State Generators (generate_start_state, generate_goal_state): Produces start and goal matrices for puzzles.
Validation (puzzle_solvable, get_inv_count): Checks if puzzles are solvable based on inversion count and empty tile row.
Solver (solve_puzzle): Implements A\* algorithm, considering heuristics to traverse nodes.
Utility Functions (swap, calc_cost, get_next_moves): Aid operations like tile swapping, cost calculation, and generating successors.

## 4. Design Decisions

Class Representation for State: Encapsulates puzzle state and associated metadata, ensuring modularity and clarity.
Global Variables for Goal State: Simplifies goal position references, enhancing lookup speed for Manhattan calculations.

## 5. Discussions

- Scalability: The current design scales with puzzle size, but performance may degrade for larger matrices due to computational and memory overhead.
- Heuristic Trade-offs: While Manhattan generally performs better than Hamming for sliding puzzles, choosing heuristics based on problem constraints could improve efficiency.
- Modularity: Functions and classes are designed modularly, making future enhancements (e.g., additional heuristics) straightforward.
- Performance Optimization: Storing visited states and open nodes in bytes minimizes memory usage.
