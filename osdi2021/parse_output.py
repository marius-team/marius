from datetime import datetime
import pandas as pd

def parse_dglke(filename):
    result = {}
    # [0]Test average MRR: 0.7992962475244683
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith("[0]Test average MRR:"):
                result["MRR"] = float(line.split()[-1])

    return result

def parse_pbg(filename):
    result = {}
    #2020-09-16 03:32:24,833   [Evaluator] Stats for edge path 1 / 1: loss:  2.3353 , pos_rank:  16.8775 , mrr:  0.826554 , r1:  0.773254 , r10:  0.912942 , r50:  0.955042 , auc:  0.985117 , count:  3449689
    with open(filename) as f:
        for line in f.readlines():
            if "Stats for edge path" in line:
                result["MRR"] = float(line.split()[16])

def read_dstat(filename):
    map_columns = {"time": "Timestamp",
                   "usr": "CPU User Utilization",
                   "sys": "CPU Sys Utilization",
                   "read.1": "Bytes Read",
                   "writ.1": "Bytes Written",
                   "used": "Memory Used"}

    df = pd.read_csv(filename, header=5).rename(columns=map_columns)
    df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%d-%m %H:%M:%S") + pd.offsets.DateOffset(years=120)
    return df


def read_nvidia_smi(filename):
    map_columns = {" utilization.memory [%]": "GPU Memory Utilization",
                   " utilization.gpu [%]": "GPU Compute Utilization",
                   "timestamp": "Timestamp"}

    df = pd.read_csv(filename).rename(columns=map_columns)
    df["Timestamp"] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M:%S.%f")
    df['GPU Memory Utilization'] = df['GPU Memory Utilization'].str.rstrip('%').astype('float') / 100.0
    df['GPU Compute Utilization'] = df['GPU Compute Utilization'].str.rstrip('%').astype('float') / 100.0
    return df


def parse_info_log(log_path):
    info_log_results = {}
    info_log_results["Train Time"] = []
    info_log_results["Shuffle Time"] = []
    info_log_results["Train Time (shuffle)"] = []
    info_log_results["Eval Time"] = []
    info_log_results["Epoch Progress"] = []
    info_log_results["Auc"] = []
    info_log_results["AvgRanks"] = []
    info_log_results["MRR"] = []
    info_log_results["Hits@1"] = []
    info_log_results["Hits@5"] = []
    info_log_results["Hits@10"] = []
    info_log_results["Hits@20"] = []
    info_log_results["Hits@50"] = []
    info_log_results["Hits@100"] = []

    info_log_results["TP"] = []
    info_log_results["TP (shuffling)"] = []

    with open(log_path) as f:

        for line in f.readlines():
            tmp = line.split("]")
            timestamp = tmp[1][2:]
            timestamp = datetime.strptime(timestamp, "%m/%d/%y %H:%M:%S.%f")
            log_string = tmp[4][1:]

            if log_string.startswith("Auc:"):
                log_string = log_string.replace(" ", "")
                kvs = log_string.split(",")

                for kv in kvs:
                    k = kv.split(":")[0]
                    v = float(kv.split(":")[1])
                    info_log_results[k].append(v)

            elif log_string.startswith("Epoch Runtime (Before shuffle/sync):") or log_string.startswith("Epoch took:"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["Train Time"].append(runtime)

            elif log_string.startswith("Shuffling Took:"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["Shuffle Time"].append(runtime)

            elif log_string.startswith("Epoch Runtime (Including shuffle/sync):"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["Train Time (shuffle)"].append(runtime)

            elif log_string.startswith("Edges per Second (Before shuffle/sync):"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["TP"].append(runtime)

            elif log_string.startswith("Edges per Second (Including shuffle/sync):"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["TP (shuffling)"].append(runtime)

            elif log_string.startswith("Evaluation complete:"):
                runtime = float(log_string.split(":")[1][1:-3])
                info_log_results["Eval Time"].append(runtime)

    info_log_results["Shuffle Time"] = [info_log_results["Train Time (shuffle)"][i] - info_log_results["Train Time"][i] for i in range(len(info_log_results["Train Time"]))]

    return info_log_results