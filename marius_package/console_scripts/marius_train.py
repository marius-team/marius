import sys

import marius as m


def main():
    print(sys.argv)
    m.marius_train(len(sys.argv), sys.argv)


if __name__ == "__main__":
    sys.exit(main())
