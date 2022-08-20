import glob
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import parsing
import ast


@dataclass
class ResultSummary:
    mean_epoch_time: int
    std_epoch_time: int

    epoch_times: list
    valid_mrr: list
    test_mrr: list

    valid_acc: list
    test_acc: list

    peak_valid_mrr: float = -1
    peak_test_mrr: float = -1

    peak_valid_acc: float = -1
    peak_test_acc: float = -1

    std_valid_mrr: float = -1
    std_test_mrr: float = -1

    mean_gpu_util: float = -1
    mean_cpu_util: float = -1
    mean_disk_util: float = -1
    std_gpu_util: float = -1
    std_cpu_util: float = -1
    std_disk_util: float = -1
    total_io: float = -1
    peak_cpu_memory_usage: float = -1
    peak_gpu_memory_usage: float = -1


def average_results(results: list):
    averaged_result = ResultSummary(-1, -1, [], [], [], [], [])
    uninitialized = True
    for result in results:

        if uninitialized:
            averaged_result.epoch_times = result.epoch_times
            averaged_result.valid_mrr = result.valid_mrr
            averaged_result.test_mrr = result.test_mrr
            averaged_result.valid_acc = result.valid_acc
            averaged_result.test_acc = result.test_acc

            averaged_result.mean_gpu_util = result.mean_gpu_util
            averaged_result.mean_cpu_util = result.mean_cpu_util
            averaged_result.mean_disk_util = result.mean_disk_util
            averaged_result.std_gpu_util = result.std_gpu_util
            averaged_result.std_cpu_util = result.std_cpu_util
            averaged_result.total_io = result.total_io
            averaged_result.peak_cpu_memory_usage = result.peak_cpu_memory_usage
            averaged_result.peak_gpu_memory_usage = result.peak_gpu_memory_usage

            uninitialized = False
        else:

            for i, val in enumerate(result.epoch_times):
                averaged_result.epoch_times[i] += val
            for i, val in enumerate(result.valid_mrr):
                averaged_result.valid_mrr[i] += val
            for i, val in enumerate(result.test_mrr):
                averaged_result.test_mrr[i] += val
            for i, val in enumerate(result.valid_acc):
                averaged_result.valid_acc[i] += val
            for i, val in enumerate(result.test_acc):
                averaged_result.test_acc[i] += val

            averaged_result.mean_gpu_util += result.mean_gpu_util
            averaged_result.mean_cpu_util += result.mean_cpu_util
            averaged_result.mean_disk_util += result.mean_disk_util
            averaged_result.std_gpu_util += result.std_gpu_util
            averaged_result.std_cpu_util += result.std_cpu_util
            averaged_result.total_io += result.total_io
            averaged_result.peak_cpu_memory_usage += result.peak_cpu_memory_usage
            averaged_result.peak_gpu_memory_usage += result.peak_gpu_memory_usage

    averaged_result.epoch_times = [t / len(results) for t in averaged_result.epoch_times]
    averaged_result.valid_mrr = [t / len(results) for t in averaged_result.valid_mrr]
    averaged_result.test_mrr = [t / len(results) for t in averaged_result.test_mrr]
    averaged_result.valid_acc = [t / len(results) for t in averaged_result.valid_acc]
    averaged_result.test_acc = [t / len(results) for t in averaged_result.test_acc]

    averaged_result.mean_gpu_util = averaged_result.mean_gpu_util / len(results)
    averaged_result.mean_cpu_util = averaged_result.mean_cpu_util / len(results)
    averaged_result.mean_disk_util = averaged_result.mean_disk_util / len(results)
    averaged_result.std_gpu_util = averaged_result.std_gpu_util / len(results)
    averaged_result.std_cpu_util = averaged_result.std_cpu_util / len(results)
    averaged_result.total_io = averaged_result.total_io / len(results)
    averaged_result.peak_cpu_memory_usage = averaged_result.peak_cpu_memory_usage / len(results)
    averaged_result.peak_gpu_memory_usage = averaged_result.peak_gpu_memory_usage / len(results)

    if len(averaged_result.valid_mrr) > 0:
        averaged_result.peak_valid_mrr = max(averaged_result.valid_mrr)
    if len(averaged_result.test_mrr) > 0:
        averaged_result.peak_test_mrr = max(averaged_result.test_mrr)

    if len(averaged_result.valid_acc) > 0:
        averaged_result.peak_valid_acc = max(averaged_result.valid_acc)

    if len(averaged_result.test_acc) > 0:
        averaged_result.peak_test_acc = max(averaged_result.test_acc)

    return averaged_result


def get_num_runs(experiment_dir: Path):
    return len(glob.glob(experiment_dir.__str__() + "/result_*"))


def get_results_summary_(experiment_dir: Path):
    num_runs = get_num_runs(experiment_dir)

    results = []
    for i in range(num_runs):
        result = ResultSummary(-1, -1, [], [], [], [], [])

        results_df = pd.read_csv(experiment_dir / Path("result_{}.csv".format(i)))
        result.epoch_times = ast.literal_eval(results_df["epoch_time"].values[0])
        result.valid_mrr = ast.literal_eval(results_df["valid_mrr"].values[0])
        result.test_mrr = ast.literal_eval(results_df["test_mrr"].values[0])
        result.valid_acc = ast.literal_eval(results_df["valid_acc"].values[0])
        result.test_acc = ast.literal_eval(results_df["test_acc"].values[0])

        dstat_df = None
        if (experiment_dir / Path("dstat_{}.csv".format(i))).exists():
            dstat_df = parsing.parse_dstat(experiment_dir / Path("dstat_{}.csv".format(i)))

            result.mean_cpu_util = (dstat_df["CPU User Utilization"] + dstat_df["CPU Sys Utilization"]).mean()
            result.mean_disk_util = (dstat_df["Bytes Read"] + dstat_df["Bytes Written"]).mean()

            result.std_cpu_util = (dstat_df["CPU User Utilization"] + dstat_df["CPU Sys Utilization"]).std()
            result.std_disk_util = (dstat_df["Bytes Read"] + dstat_df["Bytes Written"]).std()

            result.peak_memory_usage = dstat_df["Memory Used"].max()

        nvidia_smi_df = None
        if (experiment_dir / Path("nvidia_smi_{}.csv".format(i))).exists():
            nvidia_smi_df = parsing.parse_nvidia_smi(experiment_dir / Path("nvidia_smi_{}.csv".format(i)))

            result.mean_gpu_util = (nvidia_smi_df["GPU Compute Utilization"]).mean()
            result.std_gpu_util = (nvidia_smi_df["GPU Compute Utilization"]).mean()

        results.append(result)

    return average_results(results)


def print_results_summary_(experiment_dir: Path):
    results_summary = get_results_summary_(experiment_dir)

    for k, v in results_summary.__dict__.items():
        if v != -1:
            if isinstance(v, float):
                print("{}: {:.4f}".format(k, v))
            elif isinstance(v, list):
                if len(v) > 0:
                    print("{}: {}".format(k, v))


def print_results_summary(experiment_dirs: list):

    print("Printing results summary")

    for experiment_dir in experiment_dirs:
        print("-----")
        print("Experiment: {}".format(experiment_dir.__str__().split("/")[-1]))
        print_results_summary_(experiment_dir)
        print("-----")


def save_results_summary(experiment_dirs: list):
    for experiment_dir in experiment_dirs:
        print_results_summary_(experiment_dir)