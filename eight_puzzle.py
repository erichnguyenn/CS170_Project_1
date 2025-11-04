# Import future feature for type hinting compatibility
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Set
import heapq
import math
import sys

# Define a type name for readability
State = Tuple[int, ...] 

# Used for putting nodes into a priority queue (frontier)
@dataclass(order=True)
class PrioritizedItem:
    f: int           # total cost f(n) = g + h
    tie: int         # tie-breaker when f values are equal
    node: "Node" = field(compare=False)  # actual puzzle node

# A node in the search tree
@dataclass
class Node:
    state: State                
    parent: Optional["Node"]    # parent node (for backtracking path)
    action: Optional[str]       # move made to reach this node
    g: int                      # cost from start to this node
    h: int                      # heuristic estimate to goal

    # Rebuild the full path from start to this node
    def path(self) -> List["Node"]:
        n: Optional["Node"] = self
        out: List["Node"] = []
        while n is not None:
            out.append(n)
            n = n.parent
        return list(reversed(out))  # return from start → goal


# Puzzle definition and helper functions
class PuzzleProblem:
    def __init__(self, initial: State, goal: Optional[State] = None) -> None:
        self.initial = initial

        # Compute board size (k x k)
        k_float = math.sqrt(len(initial))
        if int(k_float) != k_float:
            raise ValueError("State length must be a perfect square (e.g., 9 for 8-puzzle).")
        self.k = int(k_float)

        # Set default goal state if not given
        if goal is None:
            self.goal = tuple(list(range(1, self.k*self.k)) + [0]) 
        else:
            self.goal = goal

        # Store positions of each goal tile for fast lookup
        self.goal_pos: Dict[int, Tuple[int, int]] = {}
        for idx, val in enumerate(self.goal):
            self.goal_pos[val] = (idx // self.k, idx % self.k)

    # Check if state is the goal
    def is_goal(self, s: State) -> bool:
        return s == self.goal

    # Generate valid moves from the blank position
    def neighbors(self, s: State) -> List[Tuple[str, State]]:
        k = self.k
        zero_idx = s.index(0)
        zr, zc = divmod(zero_idx, k)  # row, col of blank
        moves = []

        # Helper: swap two tiles and return new state
        def swap(idx1: int, idx2: int) -> State:
            lst = list(s)
            lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
            return tuple(lst)

        # Check possible moves and generate new states
        if zr > 0:
            moves.append(("Up", swap(zero_idx, zero_idx - k)))
        if zr < k - 1:
            moves.append(("Down", swap(zero_idx, zero_idx + k)))
        if zc > 0:
            moves.append(("Left", swap(zero_idx, zero_idx - 1)))
        if zc < k - 1:
            moves.append(("Right", swap(zero_idx, zero_idx + 1)))
        return moves

    # Heuristic: count tiles not in correct place
    def misplaced_tiles(self, s: State) -> int:
        return sum(1 for i, v in enumerate(s) if v != 0 and v != self.goal[i])

    # Heuristic: sum of Euclidean distances between tiles and their goal positions
    def euclidean_distance(self, s: State) -> int:
        total = 0.0
        k = self.k
        for idx, val in enumerate(s):
            if val == 0:
                continue  # skip blank
            r, c = divmod(idx, k)
            gr, gc = self.goal_pos[val]
            total += math.hypot(r - gr, c - gc)
        return int(total)

    # Check if puzzle is solvable using inversion count
    def is_solvable(self, s: State) -> bool:
        arr = [v for v in s if v != 0]
        inversions = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1

        # Rules differ for odd/even board size
        if self.k % 2 == 1:
            return inversions % 2 == 0
        blank_row_from_bottom = self.k - (s.index(0) // self.k)
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0


# A* search algorithm
def a_star_search(
    problem,
    heuristic: Callable[[State], int],
    trace: bool = True
) -> Tuple[Optional[Node], int, int]:
    # Start node setup
    start_h = heuristic(problem.initial)
    start = Node(state=problem.initial, parent=None, action=None, g=0, h=start_h)

    # Priority queue (frontier)
    frontier: List[PrioritizedItem] = []
    tie_counter = 0
    heapq.heappush(frontier, PrioritizedItem(f=start.g + start.h, tie=tie_counter, node=start))
    tie_counter += 1

    # Best path costs and explored states
    best_g: Dict[State, int] = {start.state: 0}
    explored: Set[State] = set()
    nodes_expanded = 0
    max_frontier_size = 1

    # Main loop
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_item = heapq.heappop(frontier)
        current = current_item.node

        if trace:
            print("\nThe best state to expand with  g(n) = {} and h(n) = {} is…".format(current.g, current.h))
            print(pretty_board(current.state, int(math.sqrt(len(current.state)))))

        # Goal test
        if problem.is_goal(current.state):
            return current, nodes_expanded, max_frontier_size

        explored.add(current.state)
        nodes_expanded += 1

        # Expand neighbors
        for action, child_state in problem.neighbors(current.state):
            new_g = current.g + 1
            if child_state in explored and new_g >= best_g.get(child_state, math.inf):
                continue
            if new_g < best_g.get(child_state, math.inf):
                best_g[child_state] = new_g
                child_h = heuristic(child_state)
                child = Node(state=child_state, parent=current, action=action, g=new_g, h=child_h)
                heapq.heappush(frontier, PrioritizedItem(f=child.g + child.h, tie=tie_counter, node=child))
                tie_counter += 1

    # No solution found
    return None, nodes_expanded, max_frontier_size


# Print puzzle board neatly
def pretty_board(state: State, k: int, blank_char: str = "b") -> str:
    rows = []
    for i in range(k):
        row = state[i*k:(i+1)*k]
        rows.append(" ".join(blank_char if v == 0 else str(v) for v in row))
    return "\n".join(rows)


# Let user type in their own puzzle
def read_puzzle_from_user(k: int) -> State:
    print("Enter your puzzle, use a zero to represent the blank.")
    data: List[int] = []
    for i in range(k):
        row = input(f"Enter row {i+1} (use spaces between numbers): ").strip()
        parts = row.split()
        if len(parts) != k:
            print(f"Expected {k} numbers; got {len(parts)}.", file=sys.stderr)
            sys.exit(1)
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            print("Non-integer value found. Aborting.", file=sys.stderr)
            sys.exit(1)
        data.extend(nums)

    # Check if valid puzzle
    if sorted(data) != list(range(k * k)):
        print(f"Input must contain all numbers 0..{k*k - 1} exactly once.", file=sys.stderr)
        sys.exit(1)
    return tuple(data)


# Main function: handles user input and runs the solver
def main():
    student_id = "XXX"
    print(f"Welcome to {student_id} 8-puzzle solver.")
    print('Type "1" to use a default puzzle, or "2" to enter your own puzzle.')
    choice = input().strip()

    if choice not in {"1", "2"}:
        print("Invalid selection.", file=sys.stderr)
        return

    # Default puzzle setup
    default_initial = (1, 2, 3,
                       4, 8, 0,
                       7, 6, 5)
    k = 3
    if choice == "1":
        initial = default_initial
    else:
        initial = read_puzzle_from_user(k)

    # Create puzzle object
    problem = PuzzleProblem(initial=initial, goal=None)

    # Choose search algorithm
    print("\nEnter your choice of algorithm")
    print("1. Uniform Cost Search")
    print("2. A* with the Misplaced Tile heuristic")
    print("3. A* with the Euclidean Distance heuristic")
    alg_choice = input().strip()

    # Select heuristic
    if alg_choice == "1":
        heuristic = (lambda s: 0)  # UCS (no heuristic)
    elif alg_choice == "2":
        heuristic = problem.misplaced_tiles
    elif alg_choice == "3":
        heuristic = problem.euclidean_distance
    else:
        print("Invalid selection.", file=sys.stderr)
        return

    # Warn if puzzle cannot be solved
    if not problem.is_solvable(problem.initial):
        print("\nWarning: This initial puzzle is not solvable. The search will run but cannot reach the goal.")

    # Run A* search
    goal_node, nodes_expanded, max_q = a_star_search(problem, heuristic, trace=True)

    # Show results
    if goal_node is not None:
        print("\nGoal!!!")
        print(f"\nTo solve this problem the search algorithm expanded a total of {nodes_expanded} nodes.")
        print(f"The maximum number of nodes in the queue at any one time: {max_q}.")
        print(f"The depth of the goal node was {goal_node.g}.")

        # Show path to goal
        path_nodes = goal_node.path()
        actions = [n.action for n in path_nodes if n.action is not None]
        if actions:
            print("\nSolution (sequence of actions):")
            print(" -> ".join(actions))
    else:
        print("\nNo solution found (frontier exhausted).")
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Max queue size: {max_q}")

if __name__ == "__main__":
    main()
