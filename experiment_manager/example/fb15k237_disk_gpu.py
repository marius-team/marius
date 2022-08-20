from marius.tools.preprocess.datasets.fb15k_237 import FB15K237
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/example/configs/")


def run_fb15k237_disk_gpu(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius, DGL, PyG
    """

    dataset_name = "fb15k237"

    marius_gs_config = BASE_PATH / Path("marius_gs_disk_gpu.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = FB15K237(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=32)
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius_gs_config, results_dir / Path("marius_gs_disk_gpu"), overwrite, enable_dstat,
                     enable_nvidia_smi, show_output, i, "marius")
