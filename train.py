from utils import Solver
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        solver = Solver.Solver(sys.argv[1])
    else:
        solver = Solver.Solver()
    solver.train()
