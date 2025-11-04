from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math

State = Tuple[int, ...] 

class PuzzleProblem:
    def __init__(self, initial: State, goal: Optional[State] = None) -> None:
        self.initial = initial
        k_float = math.sqrt(len(initial))
        if int(k_float) != k_float:
            raise ValueError("State length must be a perfect square (e.g., 9 for 8-puzzle).")
        self.k = int(k_float)
        if goal is None:
            self.goal = tuple(list(range(1, self.k*self.k)) + [0])
        else:
            self.goal = goal

        self.goal_pos: Dict[int, Tuple[int, int]] = {}
        for idx, val in enumerate(self.goal):
            self.goal_pos[val] = (idx // self.k, idx % self.k)

    def is_goal(self, s: State) -> bool:
        return s == self.goal

    def neighbors(self, s: State) -> List[Tuple[str, State]]:
        """Return list of (action, new_state) from state s. Actions are 'Up','Down','Left','Right'."""
        k = self.k
        zero_idx = s.index(0)
        zr, zc = divmod(zero_idx, k)
        moves = []
        def swap(idx1: int, idx2: int) -> State:
            lst = list(s)
            lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
            return tuple(lst)

        if zr > 0:
            moves.append(("Up", swap(zero_idx, zero_idx - k)))
        if zr < k - 1:
            moves.append(("Down", swap(zero_idx, zero_idx + k)))
        if zc > 0:
            moves.append(("Left", swap(zero_idx, zero_idx - 1)))
        if zc < k - 1:
            moves.append(("Right", swap(zero_idx, zero_idx + 1)))
        return moves

    def misplaced_tiles(self, s: State) -> int:
        return sum(1 for i, v in enumerate(s) if v != 0 and v != self.goal[i])

    def euclidean_distance(self, s: State) -> int:
        total = 0.0
        k = self.k
        for idx, val in enumerate(s):
            if val == 0:
                continue
            r, c = divmod(idx, k)
            gr, gc = self.goal_pos[val]
            total += math.hypot(r - gr, c - gc)
        return int(total)

    def is_solvable(self, s: State) -> bool:
        arr = [v for v in s if v != 0]
        inversions = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1
        if self.k % 2 == 1:
            return inversions % 2 == 0
        blank_row_from_bottom = self.k - (s.index(0) // self.k)
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0

def main():
    initial = (1, 2, 3,
               4, 8, 0,
               7, 6, 5)
    problem = PuzzleProblem(initial)
    print("Initial puzzle:")
    for action, state in problem.neighbors(initial):
        print(f"Move {action} results in:")
        print(pretty_board(state, problem.k))
    print(f"Solvable? {problem.is_solvable(initial)}")

def pretty_board(state: State, k: int, blank_char: str = "b") -> str:
    rows = []
    for i in range(k):
        row = state[i*k:(i+1)*k]
        rows.append(" ".join(blank_char if v==0 else str(v) for v in row))
    return "\n".join(rows)

if __name__ == "__main__":
    main()
