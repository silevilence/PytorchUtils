from utils.ComposedClassifier import ComposedClassifier
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        c = ComposedClassifier(config=sys.argv[1])
    else:
        c = ComposedClassifier()

    print('loading complete.')

    # a = c.classify("D:/bone/without/jin1/0.jpg")
    # print(a)
    line: str = ''
    while True:
        line = input()
        line = line.strip('\r\n')
        if line == r'exit':
            break

        if os.path.isfile(line):
            print(c.classify(line))
        else:
            print('No such file.')
