import os
import shutil
import subprocess
import sys
import time
from tracing import DstatMonitor, NvidiaSmiMonitor
import parsing
import marius as m

from pathlib import Path

CONFIG_FILE = "config_{}.yaml"
RESULT_FILE = "result_{}.csv"
OUTPUT_FILE = "out_{}.txt"

VALID_SYSTEMS = ["MARIUS", "DGL", "PYG"]


def run_command_and_pipe_output(cmd: str, output_dir: Path, run_id, output_to_terminal):
    with open(output_dir / Path(OUTPUT_FILE.format(run_id)), "w") as tmp_file:
        os.environ["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            line = line.decode("utf-8")
            if output_to_terminal:
                sys.stdout.write(line)
            tmp_file.write(line)
        proc.wait()


def run_config(config_path: Path,
               output_dir: Path,
               overwrite=True,
               enable_dstat=True,
               enable_nvidia_smi=True,
               output_to_terminal=True,
               run_id=0,
               system="marius",
               omp_num_threads=None):

    assert(config_path.exists())
    assert (system.upper() in VALID_SYSTEMS)

    os.makedirs(output_dir, exist_ok=True)
    print("=========================================")
    print("Running: {} \nConfiguration: {}\nSaving results to: {}".format(system, config_path, output_dir))

    if not overwrite and (output_dir / Path(RESULT_FILE.format(run_id))).exists():
        print("Experiment already run. Results in {}".format(output_dir / Path(RESULT_FILE.format(run_id))))
        print("=========================================")
        return
    elif overwrite and (output_dir / Path(RESULT_FILE.format(run_id))).exists():
        print("Overwriting previous experiment.")

    cmd = ""
    if system.upper() == "MARIUS":
        if omp_num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        cmd = "marius_train " + config_path.__str__()
    elif system.upper() == "DGL":
        with open(config_path, "r") as f:
            lines = f.read().splitlines()
            cmd = " ".join(lines)
    elif system.upper() == "PYG":
        with open(config_path, "r") as f:
            lines = f.read().splitlines()
            cmd = " ".join(lines)
    else:
        raise RuntimeError("Unrecognized system")

    dstat_monitor = DstatMonitor(output_dir, run_id)
    nvidia_smi_monitor = NvidiaSmiMonitor(output_dir, run_id)

    if enable_dstat:
        dstat_monitor.start()

    if enable_nvidia_smi:
        nvidia_smi_monitor.start()

    t0 = time.time()
    run_command_and_pipe_output(cmd, output_dir, run_id, output_to_terminal)
    t1 = time.time()

    shutil.copy(config_path, output_dir / Path(CONFIG_FILE.format(run_id)))

    results_df = None

    if system.upper() == "MARIUS":
        results_df = parsing.parse_marius_output(output_dir / Path(OUTPUT_FILE.format(run_id)))
    elif system.upper() == "DGL":
        results_df = parsing.parse_dgl_output(output_dir / Path(OUTPUT_FILE.format(run_id)))
    elif system.upper() == "PYG":
        results_df = parsing.parse_pyg_output(output_dir / Path(OUTPUT_FILE.format(run_id)))

    results_df.to_csv(output_dir / Path(RESULT_FILE.format(run_id)), index=False)

    if enable_dstat:
        dstat_monitor.stop()

    if enable_nvidia_smi:
        nvidia_smi_monitor.stop()

    if omp_num_threads is not None:
        del os.environ["OMP_NUM_THREADS"]

    print("Complete. Total runtime: {:.4f}s".format(t1-t0))
    print("=========================================")
