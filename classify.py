from utils.Classifier import Classifier
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        c = Classifier(config=sys.argv[1])
    else:
        c = Classifier()

    print('loading complete')

    line: str = ''
    for line in sys.stdin:
        line = line.strip('\n')
        if line == r'exit':
            break

        if os.path.isfile(line):
            print(c.classify(line))
        else:
            print('No such file.')
