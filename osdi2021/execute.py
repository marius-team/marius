import subprocess
import parse_output as p
import pathlib
import shutil

def start_tracing():
    dstat_script = """
    [ ! -e dstat_output.txt ] || rm dstat_output.txt
    dstat -t -r -c -m -d --nocolor --output dstat_output.txt > dstat_tmp.txt &
    dstat_trace_pid=\$!
    echo $dstat_trace_pid
    """

    dstat_pid = subprocess.check_output(dstat_script.split())

    nvidiasmi_script = """
    nvidia-smi %s -f nvidia_smi_output.txt &
    gpu_trace_pid=\$!
    echo $gpu_trace_pid
    """

    nvidiasmi_pid = subprocess.check_output(nvidiasmi_script.split())

    return dstat_pid, nvidiasmi_pid


def run_marius(config_path, args):
    script = """
    build/marius_train %s %s
    """

    script = script % (config_path, args)
    with open("tmp.txt", "w") as tmp_file:
        subprocess.check_call(script.split(), stdout=tmp_file)



def run_dglke(args):
    script = """
    dglke_train %s
    """ % args

    subprocess.Popen(script.split())

def run_pbg(script_path, config_path, args):
    pass


def stop_metric_collection(dstat_pid, nvidiasmi_pid):
    script = """
    kill \${%s}
    kill \${%s}
    """ % (dstat_pid, nvidiasmi_pid)

    subprocess.Popen(script.split())

def collect_metrics(info_log_only=False):
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

