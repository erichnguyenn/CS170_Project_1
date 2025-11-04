from __future__ import annotations
from typing import List, Tuple
import math

State = Tuple[int, ...]  

def chunk(state: State, k: int) -> List[List[int]]:
    return [list(state[i * k:(i + 1) * k]) for i in range(k)]

def pretty_board(state: State, k: int, blank_char: str = "b") -> str:
    rows = []
    for row in chunk(state, k):
        rows.append(" ".join(blank_char if v == 0 else str(v) for v in row))
    return "\n".join(rows)

def main(): 
    initial = (1, 2, 3,
               4, 8, 0,
               7, 6, 5)
    k = 3
    print("Initial puzzle:")
    print(pretty_board(initial, k))

if __name__ == "__main__":
    main()
