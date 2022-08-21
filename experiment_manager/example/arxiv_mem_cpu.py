from marius.tools.preprocess.datasets.ogbn_arxiv import OGBNArxiv
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/example/configs/arxiv/")


def run_ogbn_arxiv_mem_cpu(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius, DGL, PyG
    """

    dataset_name = "ogbn_arxiv"

    marius = BASE_PATH / Path("marius_gs_mem_cpu.yaml")

    dgl = BASE_PATH / Path("dgl_gs_mem_cpu.txt")

    pyg = BASE_PATH / Path("pyg_gs_mem_cpu.txt")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNArxiv(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius, results_dir / Path("ogbn_arxiv/marius_gs_mem_cpu"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")

        # Run DGL
        e.run_config(dgl, results_dir / Path("ogbn_arxiv/dgl_gs_mem_cpu"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "dgl")

        # Run PyG
        e.run_config(pyg, results_dir / Path("ogbn_arxiv/pyg_gs_mem_cpu"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "pyg")


    reporting.print_results_summary([results_dir / Path("ogbn_arxiv/marius_gs_mem_cpu"),
                                     results_dir / Path("ogbn_arxiv/dgl_gs_mem_cpu"),
                                     results_dir / Path("ogbn_arxiv/pyg_gs_mem_cpu")])
