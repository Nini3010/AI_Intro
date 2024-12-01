import heapq
import random as rd
import time
from math import sqrt

import numpy as np

# import resource


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


# create random state
puzzle_size = 9

start_state = list(range(puzzle_size))
rd.shuffle(start_state)
START_MATRIX = np.array(start_state).reshape(-1, int(sqrt(puzzle_size)))

# goal state
GOAL_MATRIX = np.array(list(range(puzzle_size))).reshape(-1, int(sqrt(puzzle_size)))


# check if state is solveable
# inversion counter
def get_inv_count(arr: np.ndarray) -> int:
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


# check if puzzle is solveable via [sum of inversions] mod 29 = 0
def puzzle_solvable(matrix: np.matrix) -> bool:
    return get_inv_count(matrix.flatten()) % 2 == 0


# heuristic 1: Hamming distance heuristic
# compare each number in matrix with goal matrix and if they differ add one to the hamming distance
# we have to exclude the blank space to get the correct distance
def get_hamming_distance(matrix: np.matrix) -> int:
    return sum(
        num != goal_num and num != 0
        for num, goal_num in zip(matrix.flatten(), GOAL_MATRIX.flatten(), strict=True)
    )


# time complexity
# space complexity


# heuristic 2: Manhattan distance
def get_manhattan_distance(matrix: np.matrix) -> int:
    distance = 0
    for row in matrix:
        for num in row:
            goal_row, goal_column = np.asarray(np.where(GOAL_MATRIX == num)).flatten()
            m_row, m_column = np.asarray(np.where(matrix == num)).flatten()
            distance += abs(m_row - goal_row) + abs(m_column - goal_column)
    return distance


# time complexity
# space complexity


def swap(matrix, row1, col1, row2, col2):
    new_matrix = matrix.copy()
    new_matrix[row1][col1], new_matrix[row2][col2] = (
        new_matrix[row2][col2],
        new_matrix[row1][col1],
    )
    return new_matrix


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


# missing: costs + current step
def calc_cost(next_move: np.matrix, heuristic: callable) -> int:
    return heuristic(next_move)


def solve_puzzle(initial_state: np.matrix) -> list[Node]:
    fastest_path: list[Node] = []
    open_and_visited_moves_bytes = []
    queue: list[(int, Node)] = []

    move = initial_state
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


# compare heuristics
print(f"start state: \n{START_MATRIX}")
solveable = puzzle_solvable(START_MATRIX)
print(f"solveable?: {solveable}")
if solveable:
    print("heuristic?: \n1: Hamming\n2: Manhattan\n9: Benchmark\n(input number)")
    heuristic_num = int(input())
    if heuristic_num == 1:
        heuristic = get_hamming_distance
    elif heuristic_num == 2:
        heuristic = get_manhattan_distance
    if heuristic is not None:
        solve_path = solve_puzzle(START_MATRIX)
        print(f"solved in {len(solve_path)} steps")
        for node in solve_path:
            print(f"step: {node.step}\n{node.board}\n")


# ram usage
# mem_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
