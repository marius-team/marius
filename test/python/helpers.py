import random
from pathlib import Path


def dataset_generator(
    train_file,
    valid_file,
    test_file,
    train_len=1000,
    valid_len=100,
    test_len=100,
    delim="\t",
    start_col=0,
    num_line_skip=0,
):
    with open(str(Path(train_file)), "w") as f:
        for i in range(num_line_skip):
            f.write("This is a line needs to be skipped.\n")
        for i in range(train_len):
            src = random.randint(1, 100)
            dst = random.randint(1, 100)
            rel = random.randint(101, 110)
            for j in range(start_col):
                f.write("col_" + str(j) + delim)
            f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
    f.close()

    with open(str(Path(valid_file)), "w") as f:
        for i in range(num_line_skip):
            f.write("This is a line needs to be skipped.\n")
        for i in range(valid_len):
            src = random.randint(1, 100)
            dst = random.randint(1, 100)
            rel = random.randint(101, 110)
            for j in range(start_col):
                f.write("col_" + str(j) + delim)
            f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
    f.close()

    with open(str(Path(test_file)), "w") as f:
        for i in range(num_line_skip):
            f.write("This is a line needs to be skipped.\n")
        for i in range(test_len):
            src = random.randint(1, 100)
            dst = random.randint(1, 100)
            rel = random.randint(101, 110)
            for j in range(start_col):
                f.write("col_" + str(j) + delim)
            f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
    f.close()
