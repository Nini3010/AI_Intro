import heapq
import random as rd
import statistics
import time
import tracemalloc

import numpy as np

GOAL_MATRIX: np.matrix
GOAL_POSITIONS: dict[int, tuple[int, int]] = {}
MEASURE: int = False


class Node:
    """
    Represents a single node in the search algorithm tree.

    Attributes:
        parent (Node): The parent node of this node.
        board (np.matrix): The current state of the puzzle represented as a matrix.
        cost (int): The cost associated with this node (heuristic value).
        step (int): The number of steps taken to reach this node.
        priority (int): The sum of cost and step, used for priority in the queue.
    """

    def __init__(self, parent, board, cost, step):
        self.parent = parent
        self.board = board
        self.cost = cost
        self.step = step
        self.priority = cost + step

    def __lt__(self, other):
        """
        Compare nodes based on priority for ordering in the priority queue.

        Args:
            other (Node): Another node to compare against.

        Returns:
            bool: True if this node's cost is less than the other.
                  If cost is the same uses board state as tie breaker.
        """
        if self.cost == other.cost:
            return self.board.tobytes() < other.board.tobytes()
        return self.cost < other.cost

    def __eq__(self, other):
        """
        Determine if two nodes are equal based on their board state, parent, and cost.

        Args:
            other (Node): The node to compare against.

        Returns:
            bool: True if the nodes have identical board states (as bytes),
                the same parent reference, and equal cost values; False otherwise.
        """
        return (self.board.tobytes(), self.parent, self.cost) == (
            other.board.tobytes(),
            other.parent,
            other.cost,
        )


def generate_start_state(puzzle_size_edge: int = 3) -> np.matrix:
    """
    Generate a random starting state as matrix for the puzzle.

    Args:
        puzzle_size_edge (int, optional): The edge length of the puzzle. Defaults to 3.

    Returns:
        np.matrix: A randomly shuffled matrix of the puzzle.
    """
    start_state = list(range(puzzle_size_edge * puzzle_size_edge))
    rd.shuffle(start_state)
    return np.array(start_state).reshape(-1, int(puzzle_size_edge))


def generate_goal_state(puzzle_size_edge: int = 3) -> None:
    """
    Generate the goal state as matrix for the puzzle, where tiles are ordered sequentially.

    Args:
        puzzle_size_edge (int, optional): The edge length of the puzzle. Defaults to 3.
    """
    global GOAL_MATRIX  # noqa: PLW0603
    GOAL_MATRIX = np.array(list(range(puzzle_size_edge * puzzle_size_edge))).reshape(
        -1, int(puzzle_size_edge)
    )


def initial_state(puzzle_size_edge: int = 3) -> np.matrix:
    """
    Initialize the puzzle by generating a random start state and setting the goal state.

    Args:
        puzzle_size_edge (int, optional): The edge length of the puzzle. Defaults to 3.

    Returns:
        np.matrix: The starting matrix for the puzzle.
    """
    # create random state
    start_state = generate_start_state(puzzle_size_edge)
    # goal state
    generate_goal_state(puzzle_size_edge)
    return start_state


def get_inv_count(arr: np.ndarray) -> int:
    """
    Count the number of inversions in a flattened puzzle matrix.\n
    (An inversion is the occurence of a higher number before a lower number,
     as this is inverse to the goal state)

    Args:
        arr (np.ndarray): The flattened puzzle matrix.

    Returns:
        int: The number of inversions.
    """
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


def puzzle_solvable(matrix: np.matrix) -> bool:
    """
    Check if the given puzzle configuration is solvable.
    This calculates differently for odd and even N in a N*N matrix.
    Both cases are covered

    Args:
        matrix (np.matrix): The puzzle matrix.

    Returns:
        bool: True if the puzzle is solvable, False otherwise.
    """
    inversions = get_inv_count(matrix.flatten())
    if matrix.shape[0] % 2 == 0:
        # for Even board size solvability: (inverions + row of empty element) needs to be odd to be solveable
        return inversions + np.argwhere(matrix == 0)[0][0] % 2 != 0
    # Even board size: odd number of inversions
    return inversions % 2 == 0


def hamming(matrix: np.matrix) -> int:
    """
    Calculate the Hamming distance of the current matrix compared to the goal state.

    Hamming distance expects that you can change every tile with every other tile freely

    Args:
        matrix (np.matrix): The current puzzle matrix.

    Returns:
        int: The Hamming distance.
    """
    # we have to exclude the blank space to get the correct distance
    # vectorised approach
    difference = (matrix != GOAL_MATRIX) & (matrix != 0)
    return np.sum(difference)


def manhattan(matrix: np.matrix) -> int:
    """
    Calculate the Manhattan distance of the current matrix compared to the goal state.

    Manhattan distance expects that you can change every tile with every neighbouring tile

    Args:
        matrix (np.matrix): The current puzzle matrix.

    Returns:
        int: The Manhattan distance.
    """

    # Cache the positions of numbers in the goal matrix
    global GOAL_POSITIONS
    if not GOAL_POSITIONS:
        for i, row in enumerate(GOAL_MATRIX):
            for j, num in enumerate(row):
                n = num.item()
                GOAL_POSITIONS[n] = (i, j)
    # loop though matrix and check against cached goal coordinates
    distance = 0
    for i, row in enumerate(matrix):
        for j, num in enumerate(row):
            if num != 0:
                n = num.item()
                goal_i, goal_j = GOAL_POSITIONS[n]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def swap(matrix: np.matrix, row1: int, col1: int, row2: int, col2: int) -> np.matrix:
    """
    Swap two tiles in the matrix and return the new matrix.

    Args:
        matrix (np.matrix): The puzzle matrix.
        row1 (int): Row index of the first tile.
        col1 (int): Column index of the first tile.
        row2 (int): Row index of the second tile.
        col2 (int): Column index of the second tile.

    Returns:
        np.matrix: A new matrix with the tiles swapped.
    """
    new_matrix = matrix.copy()
    new_matrix[row1][col1], new_matrix[row2][col2] = (
        new_matrix[row2][col2],
        new_matrix[row1][col1],
    )
    return new_matrix


def calc_cost(next_move: np.matrix, heuristic: callable) -> int:
    """
    Calculate the cost of a move using the specified heuristic.

    Args:
        next_move (np.matrix): The new puzzle matrix after a move.
        heuristic (callable): The heuristic function to calculate cost.

    Returns:
        int: The heuristic cost of the move.
    """
    return heuristic(next_move)


# generate successors: generate next possible moves
def get_next_moves(parent: Node) -> list[np.matrix]:
    """
    Generate all possible moves (successors) from the current state.

    Args:
        parent (Node): The current node.

    Returns:
        list[np.matrix]: A list of matrices representing possible moves.
    """
    next_moves = []
    empty_row, empty_column = np.argwhere(parent.board == 0)[0]
    # if row index not on top border add top as possible move
    if empty_row > 0:
        next_moves.append(
            swap(parent.board, empty_row, empty_column, empty_row - 1, empty_column)
        )
    # if row index not on bottom border add bottom as possible move
    if empty_row < parent.board.shape[0] - 1:
        next_moves.append(
            swap(parent.board, empty_row, empty_column, empty_row + 1, empty_column)
        )
    # if column index not on left border add left as possible move
    if empty_column > 0:
        next_moves.append(
            swap(parent.board, empty_row, empty_column, empty_row, empty_column - 1)
        )
    # if column index not on right border add right as possible move
    if empty_column < parent.board.shape[1] - 1:
        next_moves.append(
            swap(parent.board, empty_row, empty_column, empty_row, empty_column + 1)
        )
    return next_moves


def create_node(parent: Node, move: np.matrix, heuristic: callable) -> Node:
    """
    Create a new node for a given move.

    Args:
        parent (Node): The parent node.
        move (np.matrix): The puzzle matrix for the new move.
        heuristic (callable): The heuristic function to calculate cost.

    Returns:
        Node: The newly created node.
    """
    cost = calc_cost(move, heuristic)
    return Node(parent, move, cost, parent.step + 1)


def solve_puzzle(
    initial_matrix: np.matrix, heuristic: callable
) -> list[Node] | tuple[int, int, int]:
    """
    Solve the puzzle using the specified heuristic.

    Args:
        initial_matrix (np.matrix): The starting puzzle matrix.
        heuristic (callable): The heuristic function to guide the search.

    Returns:
        list[Node] | tuple[int, int, int]: The solution path or performance metrics.
    """
    fastest_path: list[Node] = []
    open_and_visited_moves_bytes = []
    queue: list[(int, Node)] = []
    overall_steps = 0
    expanded_nodes = 0
    queued_nodes = 0

    cost = calc_cost(initial_matrix, heuristic)
    current_node = Node(None, initial_matrix, cost, 0)
    heapq.heappush(queue, (cost, current_node))

    while queue:
        current_node: Node
        _, current_node = heapq.heappop(queue)

        open_and_visited_moves_bytes.append(current_node.board.tobytes())
        if np.array_equal(current_node.board, GOAL_MATRIX):
            if MEASURE:
                return (overall_steps, expanded_nodes, queued_nodes)
            t_node = current_node
            while t_node.parent is not None:
                fastest_path.append(t_node)
                t_node = t_node.parent
            fastest_path.reverse()
            return fastest_path

        next_moves = get_next_moves(current_node)
        expanded_nodes += len(next_moves)

        for move in next_moves:
            b_move = move.tobytes()
            if b_move not in open_and_visited_moves_bytes:
                open_and_visited_moves_bytes.append(b_move)
                node = create_node(current_node, move, heuristic)
                heapq.heappush(queue, (node.priority, node))
                queued_nodes += 1
        overall_steps += 1
    if MEASURE:
        return (overall_steps, expanded_nodes, queued_nodes)
    return fastest_path


def solve_single_puzzle(heuristics: callable, puzzle_size_edge: int) -> None:
    """
    Solve a single instance of the sliding puzzle using the chosen heuristic.

    This function generates a random solvable puzzle, determines if it can be
    solved, and then computes the solution using the specified heuristic.
    If the puzzle is unsolvable, an explanation is provided.

    Args:
        heuristics (callable): The heuristic function to guide the solution process.
        puzzle_size_edge (int): The edge length of the puzzle.

    Output:
        - Displays the starting state of the puzzle.
        - Indicates whether the puzzle is solvable and, if so, displays the
          solution steps along with the number of steps taken.
    """
    start_matrix = initial_state(puzzle_size_edge)
    print(f"start state: \n{start_matrix}")
    solveable = puzzle_solvable(start_matrix)
    print(f"inversion count: {get_inv_count(start_matrix.flatten())}")
    if not solveable and (puzzle_size_edge * puzzle_size_edge) % 2 != 0:
        print(
            """
    This matrix is NOT solveable
    Reason:
    If the board size N is an odd integer, then each legal move changes the number of inversions by an even number.
    Thus, if a board has an odd number of inversions, then it cannot lead to the goal board by a sequence of legal moves,
    because the goal board has an even number of inversions (zero).
    """
        )
    elif not solveable:
        print(
            """
    This matrix is NOT solveable
    Reason:
    The parity of the number of inversions plus the row of the blank square stays the same: each legal move changes this sum by an even number.
    If this sum is even, then it cannot lead to the goal board by a sequence of legal moves;
    if this sum is odd, then it can lead to the goal board by a sequence of legal moves, because goal board has an odd number of inversions (three).
    """
        )
    else:
        print("This matrix is solveable")
        solve_path = solve_puzzle(start_matrix, heuristics)

        print(f"solved in {len(solve_path)} steps")

        for node in solve_path:
            print(f"step: {node.step}\n{node.board}\n")


def compare_heuristics(heuristics: list[callable], puzzle_size_edge: int) -> None:
    """
    Compare the performance of multiple heuristics on a series of random puzzles.

    Args:
        heuristics (list[callable]): A list of heuristic functions to compare.
        puzzle_size_edge (int): The edge length of the puzzles.

    Output:
        - Prints detailed performance statistics for each heuristic, including:
            - Mean and standard deviation of runtime.
            - Memory usage statistics.
            - Steps taken to solve puzzles.
            - Nodes expanded and queued during the solution process.
    """
    print("Number of random boards? (default 100): ")
    board_amount = 100
    try:
        custom_board_amount = int(input() or 0)
        if custom_board_amount > 0:
            board_amount = custom_board_amount
    except ValueError:
        pass
    generate_goal_state(puzzle_size_edge)
    start_matrices = []
    while len(start_matrices) < board_amount:
        m = generate_start_state(puzzle_size_edge)
        if puzzle_solvable(m):
            start_matrices.append(m)

    heuristic_names = []
    runtimes = [[] for _ in range(len(heuristics))]
    steps = [[] for _ in range(len(heuristics))]
    expanded_nodes = [[] for _ in range(len(heuristics))]
    queued_nodes = [[] for _ in range(len(heuristics))]
    mb_used = [[] for _ in range(len(heuristics))]
    for heuristic in heuristics:
        heuristic_names.append(heuristic.__name__)
        for m in start_matrices:
            tracemalloc.start()
            start_time = time.time()
            measurements = solve_puzzle(m, heuristic)
            runtimes[len(heuristic_names) - 1].append(time.time() - start_time)
            mb_used[len(heuristic_names) - 1].append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            tracemalloc.reset_peak()
            steps[len(heuristic_names) - 1].append(measurements[0])
            expanded_nodes[len(heuristic_names) - 1].append(measurements[1])
            queued_nodes[len(heuristic_names) - 1].append(measurements[2])
    print(
        f"""
{40*"-"}
Statistics                             {heuristic_names[0]}\t\t{heuristic_names[1]}
{40*"-"}
Runtime                         mean: {statistics.mean(runtimes[0]):.2f} sec\t\t{statistics.mean(runtimes[1]):.2f} sec
                  standard deviation: {statistics.stdev(runtimes[0]):.2f} sec\t\t{statistics.stdev(runtimes[1]):.2f} sec
                      whole test run: {sum(runtimes[0])/60:.2f} mins\t\t{sum(runtimes[1])/60:.2f} mins
MB used                         mean: {statistics.mean(mb_used[0]) / (1024**2):.2f} MB\t\t{statistics.mean(mb_used[1]) / (1024**2):.2f} MB
                  standard deviation: {statistics.stdev(mb_used[0])/ (1024**2):.2f} MB\t\t{statistics.stdev(mb_used[1])/ (1024**2):.2f} MB
Algorithm complexity (steps)    mean: {statistics.mean(steps[0]):.0f}\t\t{statistics.mean(steps[1]):.0f}
Memory Effort:
Expanded Nodes                  mean: {statistics.mean(expanded_nodes[0]):.0f}\t\t{statistics.mean(expanded_nodes[1]):.0f}
Queued Nodes                    mean: {statistics.mean(queued_nodes[0]):.0f}\t\t{statistics.mean(queued_nodes[1]):.0f}
        """
    )


def main(puzzle_size_edge: int = 3) -> None:
    """
    Handle the core logic for solving the sliding puzzle game.

    Single puzzle:
        The heuristic to be used is propted via console.
        The solution steps or unsolvability explanation is displayed.

    Heuristic Comparison:
        Amount of random board as sample size to be used is propted via console.
        The metrics of each heuristic are runtime, memory usage, steps to solve
        and nodes expanded/queued.

    Args:
        puzzle_size_edge (int, optional): The edge length of the puzzle. Defaults to 3.

    Raises:
        ValueError: If an invalid heuristic choice is provided.
    """
    print(
        """
    Select the heuristic to be used (input number):
    1: Hamming
    2: Manhattan
    9: Compare all
    """
    )
    match int(input()):
        case 1:
            heuristics = hamming
        case 2:
            heuristics = manhattan
        case 9:
            heuristics = [hamming, manhattan]
            global MEASURE  # noqa: PLW0603
            MEASURE = True
        case _:
            raise ValueError("Invalid heuristic choice!")
    if MEASURE:
        compare_heuristics(heuristics, puzzle_size_edge)

    else:
        solve_single_puzzle(heuristics)


main(3)
