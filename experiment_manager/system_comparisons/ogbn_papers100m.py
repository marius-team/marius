from marius.tools.preprocess.datasets.ogbn_papers100m import OGBNPapers100M
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/system_comparisons/configs/ogbn_papers100m")


def run_ogbn_papers100m(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius, DGL, PyG
    """

    dataset_name = "ogbn_papers100m"

    marius_gs = BASE_PATH / Path("marius_gs.yaml")

    dgl_gs = BASE_PATH / Path("dgl_gs.txt")

    pyg_gs = BASE_PATH / Path("pyg_gs.txt")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNPapers100M(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius_gs, results_dir / Path("ogbn_papers100m/marius_gs"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius", omp_num_threads=8)

        # Run DGL
        e.run_config(dgl_gs, results_dir / Path("ogbn_papers100m/dgl_gs"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "dgl")

        # Run PyG
        e.run_config(pyg_gs, results_dir / Path("ogbn_papers100m/pyg_gs"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "pyg")

    reporting.print_results_summary([results_dir / Path("ogbn_papers100m/marius_gs"),
                                     results_dir / Path("ogbn_papers100m/dgl_gs"),
                                     results_dir / Path("ogbn_papers100m/pyg_gs")])
