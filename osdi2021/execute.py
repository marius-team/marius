import pathlib
import shutil
import subprocess
import sys
import parse_output as p
import os


def start_tracing():
    dstat_script = """
    dstat -t -r -c -m -d --nocolor --output dstat_output.txt
    """

    try:
        shutil.rmtree("dstat_output.txt")
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree("dstat_tmp.txt")
    except FileNotFoundError:
        pass

    with open("dstat_tmp.txt", "w") as f:
        dstat_pid = subprocess.Popen(dstat_script.split(), stdout=f).pid

    print("Started dstat")

    try:
        shutil.rmtree("nvidia_smi_output.txt")
    except FileNotFoundError:
        pass

    nvidiasmi_script = """
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f nvidia_smi_output.txt
    """

    nvidiasmi_pid = subprocess.Popen(nvidiasmi_script.split()).pid

    print("Started nvidia-smi")

    return dstat_pid, nvidiasmi_pid


def run_marius(config_path, args, show_output=False):
    script = """
    marius_train %s %s
    """

    script = script % (config_path, args)
    with open("tmp.txt", "w") as tmp_file:
        proc = subprocess.Popen(script.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            line = line.decode("utf-8")
            if show_output:
                sys.stdout.write(line)
            tmp_file.write(line)
        proc.wait()


def run_dglke(args, show_output=False):
    with open("tmp.txt", "w") as tmp_file:
        os.environ["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(args.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            line = line.decode("utf-8")
            if show_output:
                sys.stdout.write(line)
            tmp_file.write(line)
        proc.wait()


def run_pbg(script_path, config_path, args=None, show_output=False):
    script = "python3 %s --config %s" % (script_path, config_path)
    with open("tmp.txt", "w") as tmp_file:
        os.environ["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(script.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            line = line.decode("utf-8")
            if show_output:
                sys.stdout.write(line)
            tmp_file.write(line)
        proc.wait()


def stop_metric_collection(dstat_pid, nvidiasmi_pid):
    try:
        script = """
        kill %s
        """ % dstat_pid
        subprocess.check_call(script, shell=True)
    except Exception as e:
        print("Unable to kill dstat: %s" % e)

    try:
        script = """
        kill %s
        """ % nvidiasmi_pid
        subprocess.check_call(script, shell=True)
    except Exception as e:
        print("Unable to kill nvidia-smi: %s" % e)


def collect_metrics(info_log_only=False, dglke=False, pbg=False):
    if dglke:
        info_log = p.parse_dglke("tmp.txt")
    elif pbg:
        info_log = p.parse_pbg("tmp.txt")
    else:
        info_log = p.parse_info_log("logs/marius_info.log")

    dstat_df = None
    nvidia_smi_df = None
    if info_log_only is False:
        dstat_df = p.read_dstat("dstat_output.txt")
        nvidia_smi_df = p.read_nvidia_smi("nvidia_smi_output.txt")

    return info_log, dstat_df, nvidia_smi_df


def cleanup_experiments(info_log_only=False):
    if info_log_only is False:
        try:
            pathlib.Path.unlink(pathlib.Path("dstat_output.txt"))
        except FileNotFoundError:
            pass
        try:
            pathlib.Path.unlink(pathlib.Path("nvidia_smi_output.txt"))
        except FileNotFoundError:
            pass
        try:
            pathlib.Path.unlink(pathlib.Path("dstat_tmp.txt"))
        except FileNotFoundError:
            pass

    try:
        pathlib.Path.unlink(pathlib.Path("tmp.txt"))
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree("logs")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("training_dir")
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree("ckpts")
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree("model")
    except FileNotFoundError:
        pass

