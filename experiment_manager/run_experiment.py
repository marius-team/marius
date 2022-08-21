import argparse
import os
from pathlib import Path

from setup_dgl.setup_dgl import run_setup_dgl
from example.fb15k237_disk_cpu import run_fb15k237_disk_cpu
from example.fb15k237_disk_gpu import run_fb15k237_disk_gpu
from example.fb15k237_mem_cpu import run_fb15k237_mem_cpu
from example.fb15k237_mem_gpu import run_fb15k237_mem_gpu

# from system_comparisons.fb15k_237 import run_fb15k_237
# from system_comparisons.livejournal import run_livejournal
# from system_comparisons.twitter import run_twitter
# from system_comparisons.freebase86m import run_freebase86m
# from system_comparisons.ogbn_arxiv import run_ogbn_arxiv
# from system_comparisons.ogbn_papers100m import run_ogbn_papers100m
# from system_comparisons.ogb_wikikg90mv2 import run_ogb_wikikg90mv2
# from system_comparisons.ogbl_citation2 import run_ogbl_citation2
# from system_comparisons.ogbl_ppa import run_ogbl_ppa
# from system_comparisons.ogbl_wikikg2 import run_ogbl_wikikg2
# from system_comparisons.ogbn_products import run_ogbn_products
# from system_comparisons.ogb_mag240m import run_ogb_mag240m
# from system_comparisons.ogbn_papers100m_seq import run_ogbn_papers100m_seq
# from system_comparisons.ogb_mag240m_seq import run_ogb_mag240m_seq
# from system_comparisons.ogb_wikikg90mv2_disk import run_ogb_wikikg90mv2_disk
# from system_comparisons.freebase86m_disk import run_freebase86m_disk
# from sota.featurizer import run_featurizer
# from sampling.train_trace import run_train_trace
# from sampling.sampling import run_sampling_obgn_papers100m
# from disk.ogbn_papers100m_disk import run_ogbn_papers100m_disk
# from disk.freebase86m_beta_battles import run_freebase86m_beta_battles
# from disk.fb15k237 import run_fb15k237_disk
# from disk.wiki import run_wiki_beta_battles
# from disk.freebase86m_scans import run_freebase86m_scans

DEFAULT_DATASET_DIRECTORY = "datasets/"
DEFAULT_RESULTS_DIRECTORY = "results/"

if __name__ == "__main__":
    experiment_dict = {
        "setup_dgl": run_setup_dgl,
        "fb15k237_disk_cpu": run_fb15k237_disk_cpu,
        "fb15k237_disk_gpu": run_fb15k237_disk_gpu,
        "fb15k237_mem_cpu": run_fb15k237_mem_cpu,
        "fb15k237_mem_gpu": run_fb15k237_mem_gpu
        # "fb15k_237": run_fb15k_237,
        # "livejournal": run_livejournal,
        # "freebase86m": run_freebase86m,
        # "ogbn_arxiv": run_ogbn_arxiv,
        # "twitter": run_twitter,
        # "ogbn_papers100m": run_ogbn_papers100m,
        # "ogb_wikikg90mv2": run_ogb_wikikg90mv2,
        # "ogbl_citation2": run_ogbl_citation2,
        # "ogbl_ppa": run_ogbl_ppa,
        # "ogbl_wikikg2": run_ogbl_wikikg2,
        # "ogbn_products": run_ogbn_products,
        # "ogb_mag240m": run_ogb_mag240m,
        # "ogbn_papers100m_seq": run_ogbn_papers100m_seq,
        # "ogb_mag240m_seq": run_ogb_mag240m_seq,
        # "ogb_wikikg90mv2_disk": run_ogb_wikikg90mv2_disk,
        # "freebase86m_disk": run_freebase86m_disk,
        # "ogbn_papers100m_disk": run_ogbn_papers100m_disk,
        # "freebase86m_beta_battles": run_freebase86m_beta_battles,
        # "fb15k237_disk": run_fb15k237_disk,
        # "wiki_beta_battles": run_wiki_beta_battles,
        # "freebase86m_scans": run_freebase86m_scans,
        # "training_trace": run_train_trace,
        # "sampling_papers100m": run_sampling_obgn_papers100m,
        # "featurizer": run_featurizer
    }

    parser = argparse.ArgumentParser(description='Reproduce experiments ')
    parser.add_argument('--experiment', metavar='experiment', type=str, choices=experiment_dict.keys(),
                        help='Experiment choices: %(choices)s')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='If true, the results of previously run experiments will be overwritten.')
    parser.add_argument('--enable_dstat', dest='enable_dstat', action='store_true',
                        help='If true, dstat resource utilization metrics.')
    parser.add_argument('--enable_nvidia_smi', dest='enable_nvidia_smi', action='store_true',
                        help='If true, nvidia-smi will collect gpu utilization metrics.')
    parser.add_argument('--dataset_dir', metavar='dataset_dir', type=str, default=DEFAULT_DATASET_DIRECTORY,
                        help='Directory containing preprocessed dataset(s). If a given dataset is not present'
                             ' then it will be downloaded and preprocessed in this directory')
    parser.add_argument('--results_dir', metavar='results_dir', type=str, default=DEFAULT_RESULTS_DIRECTORY,
                        help='Directory for output of results')
    parser.add_argument('--show_output', dest='show_output', action='store_true',
                        help='If true, the output of each run will be printed directly to the terminal.')
    parser.add_argument('--short', dest='short', action='store_true',
                        help='If true, a shortened version of the experiment(s) will be run')
    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1,
                        help='Number of runs for each configuration. Used to average results.')

    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.results_dir = Path(args.results_dir)

    if not args.dataset_dir.exists():
        os.makedirs(args.dataset_dir)
    if not args.results_dir.exists():
        os.makedirs(args.results_dir)

    experiment_dict.get(args.experiment)(args.dataset_dir,
                                         args.results_dir,
                                         args.overwrite,
                                         args.enable_dstat,
                                         args.enable_nvidia_smi,
                                         args.show_output,
                                         args.short,
                                         args.num_runs)
