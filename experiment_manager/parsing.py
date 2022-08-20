from pathlib import Path
import pandas as pd
import re
from typing import Optional, Type


def get_number_from_line(line, number_idx, mode: Optional[Type] = int):
    numbers = re.findall(r"[-e.0-9]*", line)
    numbers = [x for x in numbers if (len(x) > 0 and x != 'e')]
    if mode == int:
        return int(numbers[number_idx])
    elif mode == float:
        return float(numbers[number_idx])
    else:
        raise Exception("Invalid number type requested.")


def parse_marius_output(input_file: Path) -> pd.DataFrame:

    assert(input_file.exists())

    dict_results = {
        "init_time": [-1],
        "epoch_time": [[]],
        "valid_acc": [[]],
        "valid_mr": [[]],
        "valid_mrr": [[]],
        "valid_hits1": [[]],
        "valid_hits3": [[]],
        "valid_hits5": [[]],
        "valid_hits10": [[]],
        "valid_hits50": [[]],
        "valid_hits100": [[]],
        "test_acc": [[]],
        "test_mr": [[]],
        "test_mrr": [[]],
        "test_hits1": [[]],
        "test_hits3": [[]],
        "test_hits5": [[]],
        "test_hits10": [[]],
        "test_hits50": [[]],
        "test_hits100": [[]],
    }

    valid = True

    with open(input_file, "r") as f:
        for line in f.readlines():
            if "Initialization" in line:
                dict_results["init_time"][0] = float(line.split()[-1][:-1])

            if "Epoch Runtime" in line:
                dict_results["epoch_time"][0].append(float(line.split()[-1][:-2]) / 1000.0)

            if "Evaluating validation set" in line:
                valid = True

            if "Evaluating test set" in line:
                valid = False

            if "Accuracy" in line:
                if valid:
                    dict_results["valid_acc"][0].append(float(line.split()[-1][:-1]))
                else:
                    dict_results["test_acc"][0].append(float(line.split()[-1][:-1]))

            if "Mean Rank" in line:
                if valid:
                    dict_results["valid_mr"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_mr"][0].append(float(line.split()[-1]))

            if "MRR" in line:
                if valid:
                    dict_results["valid_mrr"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_mrr"][0].append(float(line.split()[-1]))

            if "Hits@1:" in line:
                if valid:
                    dict_results["valid_hits1"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits1"][0].append(float(line.split()[-1]))

            if "Hits@3:" in line:
                if valid:
                    dict_results["valid_hits3"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits3"][0].append(float(line.split()[-1]))

            if "Hits@5:" in line:
                if valid:
                    dict_results["valid_hits5"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits5"][0].append(float(line.split()[-1]))

            if "Hits@10:" in line:
                if valid:
                    dict_results["valid_hits10"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits10"][0].append(float(line.split()[-1]))

            if "Hits@50:" in line:
                if valid:
                    dict_results["valid_hits50"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits50"][0].append(float(line.split()[-1]))

            if "Hits@100:" in line:
                if valid:
                    dict_results["valid_hits100"][0].append(float(line.split()[-1]))
                else:
                    dict_results["test_hits100"][0].append(float(line.split()[-1]))

    return pd.DataFrame(dict_results)


def parse_dgl_output(input_file: Path) -> pd.DataFrame:
    assert(input_file.exists())

    dict_results = {
        "init_time": [-1],
        "epoch_time": [[]],
        "valid_acc": [[]],
        "valid_mr": [[]],
        "valid_mrr": [[]],
        "valid_hits1": [[]],
        "valid_hits3": [[]],
        "valid_hits5": [[]],
        "valid_hits10": [[]],
        "valid_hits50": [[]],
        "valid_hits100": [[]],
        "test_acc": [[]],
        "test_mr": [[]],
        "test_mrr": [[]],
        "test_hits1": [[]],
        "test_hits3": [[]],
        "test_hits5": [[]],
        "test_hits10": [[]],
        "test_hits50": [[]],
        "test_hits100": [[]],
    }

    with open(input_file, "r") as f:
        for line in f.readlines():

            if "training time: " in line:
                dict_results["epoch_time"][0].append(get_number_from_line(line, 1, float))

            if "Accuracy for " in line:
                if "valid" in line:
                    dict_results["valid_acc"][0].append(get_number_from_line(line, 0, float))
                else:
                    dict_results["test_acc"][0].append(get_number_from_line(line, 0, float))

            if "MRR for " in line:
                if "valid" in line:
                    dict_results["valid_mrr"][0].append(get_number_from_line(line, 0, float))
                else:
                    dict_results["test_mrr"][0].append(get_number_from_line(line, 0, float))

            if "Hits@1 for " in line:
                if "valid" in line:
                    dict_results["valid_hits1"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits1"][0].append(get_number_from_line(line, 1, float))

            if "Hits@3:" in line:
                if "valid" in line:
                    dict_results["valid_hits3"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits3"][0].append(get_number_from_line(line, 1, float))

            if "Hits@5:" in line:
                if "valid" in line:
                    dict_results["valid_hits5"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits5"][0].append(get_number_from_line(line, 1, float))

            if "Hits@10:" in line:
                if "valid" in line:
                    dict_results["valid_hits10"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits10"][0].append(get_number_from_line(line, 1, float))

            if "Hits@50:" in line:
                if "valid" in line:
                    dict_results["valid_hits50"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits50"][0].append(get_number_from_line(line, 1, float))

            if "Hits@100:" in line:
                if "valid" in line:
                    dict_results["valid_hits100"][0].append(get_number_from_line(line, 1, float))
                else:
                    dict_results["test_hits100"][0].append(get_number_from_line(line, 1, float))

    return pd.DataFrame(dict_results)


def parse_pyg_output(input_file: Path) -> pd.DataFrame:

    assert(input_file.exists())

    dict_results = {
        "init_time": [-1],
        "epoch_time": [[]],
        "valid_acc": [[]],
        "valid_mr": [[]],
        "valid_mrr": [[]],
        "valid_hits1": [[]],
        "valid_hits3": [[]],
        "valid_hits5": [[]],
        "valid_hits10": [[]],
        "valid_hits50": [[]],
        "valid_hits100": [[]],
        "test_acc": [[]],
        "test_mr": [[]],
        "test_mrr": [[]],
        "test_hits1": [[]],
        "test_hits3": [[]],
        "test_hits5": [[]],
        "test_hits10": [[]],
        "test_hits50": [[]],
        "test_hits100": [[]],
    }

    with open(input_file, "r") as f:
        for line in f.readlines():
            if "Initialization" in line:
                dict_results["init_time"][0] = float(line.split()[-1][:-1])

            if "Training took" in line:
                dict_results["epoch_time"][0].append(float(line.split()[-1][:-1]))

            if "VALID/TEST MRR:" in line:
                dict_results["valid_mrr"][0].append(float(line.split()[-1].split("/")[0]))
                dict_results["test_mrr"][0].append(float(line.split()[-1].split("/")[1]))

            if "VALID/TEST Acc:" in line:
                dict_results["valid_acc"][0].append(float(line.split()[-1].split("/")[0]))
                dict_results["test_acc"][0].append(float(line.split()[-1].split("/")[1]))

    return pd.DataFrame(dict_results)


def parse_dstat(input_file: Path):
    map_columns = {"time": "Timestamp",
                   "usr": "CPU User Utilization",
                   "sys": "CPU Sys Utilization",
                   "read.1": "Bytes Read",
                   "writ.1": "Bytes Written",
                   "used": "Memory Used"}

    df = pd.read_csv(input_file, header=5).rename(columns=map_columns)
    df = df[:-1] # last line might not have been completely written, ignore it
    df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%d-%m %H:%M:%S") + pd.offsets.DateOffset(years=120)
    return df


def parse_nvidia_smi(input_file: Path):
    map_columns = {" utilization.memory [%]": "GPU Memory Utilization",
                   " utilization.gpu [%]": "GPU Compute Utilization",
                   " memory.used [MiB]": "GPU Memory Used",
                   " timestamp": "Timestamp"}

    df = pd.read_csv(input_file).rename(columns=map_columns)
    df = df[:-1] # last line might not have been completely written, ignore it
    df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M:%S.%f")
    df['GPU Memory Utilization'] = df['GPU Memory Utilization'].str.rstrip('%').astype('float') / 100.0
    df['GPU Compute Utilization'] = df['GPU Compute Utilization'].str.rstrip('%').astype('float') / 100.0
    df['GPU Memory Used'] = df['GPU Memory Used'].str.rstrip(" MiB").astype('float')
    return df

