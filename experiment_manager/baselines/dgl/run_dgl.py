import argparse

from node_classification import run_nc
from link_prediction import run_lp


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # [dataset_options]
    p.add_argument("--base_directory")
    p.add_argument("--learning_task", default="link_prediction")

    # [emb_options]
    p.add_argument("--emb_dim", type=int, default=50)
    p.add_argument("--emb_storage_device", type=str, default="CPU")
    p.add_argument("--emb_storage_backend", type=str, default="dgl_sparse")

    # [encoder/decoder_options]
    p.add_argument("--encode", type=str2bool, default=True) # not necessarily optimized for no encode
    p.add_argument("--model", default="graph_sage")
    p.add_argument("--outgoing_nbrs", type=str2bool, default=True)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--h_dim", type=int, default=50)
    p.add_argument("--out_dim", type=int, default=50)

    # [model_options]
    p.add_argument("--graph_sage_aggregator", default="mean")
    p.add_argument("--graph_sage_dropout", type=float, default=0.0)
    p.add_argument("--gat_num_heads", type=int, default=10)
    p.add_argument("--gat_feat_drop", type=float, default=0.0)
    p.add_argument("--gat_attn_drop", type=float, default=0.0)
    p.add_argument("--gat_negative_slope", type=float, default=0.2)

    # [train_options]
    p.add_argument("--train_batch_size", type=int, default=1000)
    p.add_argument("--single_format", type=str2bool, default=False)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--sample_device", type=str, default="CPU")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="Adagrad")
    p.add_argument("--num_train_nbrs", nargs="+", type=int, default=[-1])
    p.add_argument("--num_train_chunks", type=int, default=10)
    p.add_argument("--num_train_uniform_negs", type=int, default=1000)
    p.add_argument("--num_train_deg_negs", type=int, default=0)

    # [eval_options]
    p.add_argument("--epochs_per_eval", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1000)
    p.add_argument("--filtered_mrr", type=str2bool, default=False)
    p.add_argument("--num_eval_nbrs", nargs="+", type=int, default=[-1])
    p.add_argument("--num_eval_chunks", type=int, default=10)
    p.add_argument("--num_eval_uniform_negs", type=int, default=1000)
    p.add_argument("--num_eval_deg_negs", type=int, default=0)

    # [performance_options]
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--persistent_workers", type=str2bool, default=False)

    p.add_argument('--print_timing', action='store_true', default=False)
    p.add_argument('--only_sample', action='store_true', default=False)
    p.add_argument('--no_compute', action='store_true', default=False)

    config_args = p.parse_args()

    if config_args.learning_task.upper() == "LINK_PREDICTION":
        run_lp(config_args)
    else:
        run_nc(config_args)

