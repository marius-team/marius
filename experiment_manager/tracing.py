import os
import subprocess
from pathlib import Path
import shutil

DSTAT_OUTPUT_FILE = "dstat_{}.csv"
NVIDIA_SMI_OUTPUT_FILE = "nvidia_smi_{}.csv"

dstat_cmd = """
            dstat -t -r -c -m -d --nocolor --output {}
            """

nvidia_smi_cmd = """
                 nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f {}
                 """


class Monitor(object):
    def __init__(self, output_dir: Path, run_id: int):
        self.output_dir = output_dir
        self.run_id = run_id


class DstatMonitor(Monitor):
    def start(self):

        output_file = self.output_dir / Path(DSTAT_OUTPUT_FILE.format(self.run_id))
        if output_file.exists():
            os.unlink(output_file)

        cmd = dstat_cmd.format(output_file.__str__())
        with open(os.devnull, "w") as f:
            self.pid = subprocess.Popen(cmd.split(), stdout=f).pid

    def stop(self):
        try:
            os.kill(self.pid, 9)
        except Exception as e:
            print("Unable to kill dstat: %s" % e)


class NvidiaSmiMonitor(Monitor):

    def start(self):
        output_file = (self.output_dir / Path(NVIDIA_SMI_OUTPUT_FILE.format(self.run_id)))
        if output_file.exists():
            os.unlink(output_file)

        cmd = nvidia_smi_cmd.format(output_file.__str__())
        self.pid = subprocess.Popen(cmd.split()).pid

    def stop(self):
        try:
            os.kill(self.pid, 9)
        except Exception as e:
            print("Unable to kill nvidia-smi: %s" % e)
