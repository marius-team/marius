import pathlib
import shutil
import subprocess

import parse_output as p


def start_tracing():
    dstat_script = """
    dstat -t -r -c -m -d --nocolor --output dstat_output.txt
    """

    with open("dstat_tmp.txt", "w") as f:
        dstat_pid = subprocess.Popen(dstat_script.split(), stdout=f).pid

    nvidiasmi_script = """
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f nvidia_smi_output.txt
    """

    nvidiasmi_pid = subprocess.Popen(nvidiasmi_script.split()).pid

    return dstat_pid, nvidiasmi_pid


def run_marius(config_path, args):
    script = """
    build/marius_train %s %s
    """

    script = script % (config_path, args)
    with open("tmp.txt", "w") as tmp_file:
        subprocess.check_call(script, shell=True, stdout=tmp_file)


def run_dglke(args):
    with open("tmp.txt", "w") as tmp_file:
        subprocess.check_call(args, shell=True, stdout=tmp_file)


def run_pbg(script_path, config_path, args=None):
    script = "%s --config %s" % (script_path, config_path)
    with open("tmp.txt", "w") as tmp_file:
        subprocess.check_call(script, shell=True, stdout=tmp_file)


def stop_metric_collection(dstat_pid, nvidiasmi_pid):
    script = """
    kill %s
    kill %s
    """ % (dstat_pid, nvidiasmi_pid)

    subprocess.check_call(script, shell=True)


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
