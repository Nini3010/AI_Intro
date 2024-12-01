import heapq
import random as rd
import statistics
import time

import numpy as np

# import math
# import resource
GOAL_MATRIX: np.matrix


class Node:
    def __init__(self, parent, board, cost, step):
        self.parent = parent
        self.board = board
        self.cost = cost
        self.step = step
        self.priority = cost + step

    def __lt__(self, other):
        return (self.board.tobytes(), self.step) < (other.board.tobytes(), other.step)

    def __eq__(self, other):
        return (self.board.tobytes(), self.step) == (other.board.tobytes(), self.step)


def generate_start_matrix(puzzle_size_edge: int = 3) -> np.matrix:
    start_state = list(range(puzzle_size_edge * puzzle_size_edge))
    rd.shuffle(start_state)
    return np.array(start_state).reshape(-1, int(puzzle_size_edge))


def generate_goal_matrix(puzzle_size_edge: int = 3) -> None:
    global GOAL_MATRIX  # noqa: PLW0603
    GOAL_MATRIX = np.array(list(range(puzzle_size_edge * puzzle_size_edge))).reshape(
        -1, int(puzzle_size_edge)
    )


def initial_state(puzzle_size_edge: int = 3) -> np.matrix:
    # create random state
    start_matrix = generate_start_matrix(puzzle_size_edge)
    # goal state
    generate_goal_matrix()
    return start_matrix


# inversion counter
def get_inv_count(arr: np.ndarray) -> int:
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


# check if puzzle is solveable via [sum of inversions] mod 2 = 0
def puzzle_solvable(matrix: np.matrix) -> bool:
    inversions = get_inv_count(matrix.flatten())
    if matrix.shape[0] % 2 == 0:
        # for Even board size solvability: (inverions + row of empty element) needs to be odd to be solveable
        return (inversions + np.asarray(np.where(matrix == 0)).flatten()[0] % 2 != 0,)
    # Even board size: odd number of inversions
    return inversions % 2 == 0


# Hamming distance heuristic
# can change every tile with every other tile
# compare each number in matrix with goal matrix and if they differ add one to the hamming distance
# we have to exclude the blank space to get the correct distance
def hamming(matrix: np.matrix) -> int:
    return sum(
        num != goal_num and num != 0
        for num, goal_num in zip(matrix.flatten(), GOAL_MATRIX.flatten(), strict=True)
    )


# heuristic 2: Manhattan distance
# can change every tile with every neighbouring tile
def manhattan(matrix: np.matrix) -> int:
    distance = 0
    for row in matrix:
        for num in row:
            goal_row, goal_column = np.asarray(np.where(GOAL_MATRIX == num)).flatten()
            m_row, m_column = np.asarray(np.where(matrix == num)).flatten()
            distance += abs(m_row - goal_row) + abs(m_column - goal_column)
    return distance


def swap(matrix: np.matrix, row1: int, col1: int, row2: int, col2: int) -> np.matrix:
    new_matrix = matrix.copy()
    new_matrix[row1][col1], new_matrix[row2][col2] = (
        new_matrix[row2][col2],
        new_matrix[row1][col1],
    )
    return new_matrix


# missing: costs + current step
def calc_cost(next_move: np.matrix, heuristic: callable) -> int:
    return heuristic(next_move)


# generate successors: generate next possible moves
def get_child_nodes(parent: Node, heuristic: callable) -> list[Node]:
    parent_matrix = parent.board
    next_moves = []
    empty_row, empty_column = np.asarray(np.where(parent_matrix == 0)).flatten()
    # if row index not on top border add top as possible move
    if empty_row > 0:
        next_moves.append(
            swap(parent_matrix, empty_row, empty_column, empty_row - 1, empty_column)
        )
    # if row index not on bottom border add bottom as possible move
    if empty_row < parent_matrix.shape[0] - 1:
        next_moves.append(
            swap(parent_matrix, empty_row, empty_column, empty_row + 1, empty_column)
        )
    # if column index not on left border add left as possible move
    if empty_column > 0:
        next_moves.append(
            swap(parent_matrix, empty_row, empty_column, empty_row, empty_column - 1)
        )
    # if column index not on right border add right as possible move
    if empty_column < parent_matrix.shape[1] - 1:
        next_moves.append(
            swap(parent_matrix, empty_row, empty_column, empty_row, empty_column + 1)
        )

    child_nodes = []
    for move in next_moves:
        cost = calc_cost(move, heuristic)
        child = Node(parent, move, cost, parent.step + 1)
        child_nodes.append(child)
    return child_nodes


def solve_puzzle(initial_matrix: np.matrix, heuristic: callable) -> list[Node]:
    fastest_path: list[Node] = []
    open_and_visited_moves_bytes = []
    queue: list[(int, Node)] = []

    move = initial_matrix
    step = 0
    cost = calc_cost(move, heuristic)
    priority = cost
    node = Node(None, move, cost, step)
    heapq.heappush(queue, (priority, node))

    while queue:
        heap_op = heapq.heappop(queue)
        priority: int = heap_op[0]
        node: Node = heap_op[1]

        # visited_moves.append((priority, cost, step, move))
        open_and_visited_moves_bytes.append(node.board.tobytes())
        if np.array_equal(node.board, GOAL_MATRIX):
            t_node = node
            while t_node.parent is not None:
                fastest_path.append(t_node)
                t_node = t_node.parent
            fastest_path.reverse()
            return fastest_path

        child_nodes = get_child_nodes(node, heuristic)

        for node in child_nodes:
            b_move = node.board.tobytes()
            if b_move not in open_and_visited_moves_bytes:
                open_and_visited_moves_bytes.append(b_move)
                heapq.heappush(queue, (node.priority, node))
    return fastest_path


def main():
    puzzle_size_edge = 3
    measure = False
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
            measure = True
        case _:
            raise ValueError("Invalid heuristic choice!")
    if measure:
        print("Number of random boards? (default 100): ")
        board_amount = 100
        try:
            board_amount = int(input() or 0)
        except ValueError:
            pass
        generate_goal_matrix(puzzle_size_edge)
        start_matrices = []
        while len(start_matrices) < board_amount:
            m = generate_start_matrix(puzzle_size_edge)
            if puzzle_solvable(m):
                start_matrices.append(m)

        heuristic_names = []
        runtimes = [[] for _ in range(len(heuristics))]
        for heuristic in heuristics:
            heuristic_names.append(heuristic.__name__)
            for m in start_matrices:
                start_time = time.time()
                solve_puzzle(m, heuristic)
                runtimes[len(heuristic_names) - 1].append(time.time() - start_time)
        print(
            f"""
{40*"-"}
Statistics          {heuristic_names[0]}        {heuristic_names[1]}
{40*"-"}
Runtime(sec)    mean: {statistics.mean(runtimes[0])}        {statistics.mean(runtimes[1])}
            """
        )

    else:
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


main()
# ram usage
# mem_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# mem_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
