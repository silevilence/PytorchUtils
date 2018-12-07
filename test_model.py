from utils import Tester
import sys

if __name__ == "__main__":
    if len(sys.argv) > 2:
        tester = Tester.Tester(sys.argv[1], sys.argv[2])
    else:
        tester = Tester.Tester()
    tester.test()