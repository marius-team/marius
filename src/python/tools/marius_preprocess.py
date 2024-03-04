import argparse
import shutil
from pathlib import Path

from marius.tools.preprocess import custom
from marius.tools.preprocess.datasets import (
    fb15k,
    fb15k_237,
    freebase86m,
    livejournal,
    ogb_mag240m,
    ogb_wikikg90mv2,
    ogbl_citation2,
    ogbl_collab,
    ogbl_ppa,
    ogbl_wikikg2,
    ogbn_arxiv,
    ogbn_papers100m,
    ogbn_products,
    twitter,
)


def set_args():
    parser = argparse.ArgumentParser(description="Preprocess Datasets", prog="preprocess")

    parser.add_argument(
        "--output_directory", metavar="output_directory", type=str, default="", help="Directory to put graph data"
    )

    parser.add_argument(
        "--edges", metavar="edges", nargs="+", type=str, help="File(s) containing the edge list(s) for a custom dataset"
    )

    parser.add_argument(
        "--dataset", metavar="dataset", type=str, default="custom", help="Name of dataset to preprocess"
    )

    parser.add_argument(
        "--num_partitions",
        metavar="num_partitions",
        required=False,
        type=int,
        default=1,
        help="Number of node partitions",
    )

    parser.add_argument(
        "--partitioned_eval",
        action="store_true",
        default=False,
        help="If true, the validation and/or the test set will be partitioned.",
    )

    parser.add_argument(
        "--delim", "-d", metavar="delim", type=str, default="\t", help="Delimiter to use for delimited file inputs"
    )

    parser.add_argument(
        "--dataset_split",
        "-ds",
        metavar="dataset_split",
        nargs="+",
        type=float,
        default=None,
        help="Split dataset into specified fractions",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If true, the preprocessed dataset will be overwritten if it already exists",
    )

    parser.add_argument(
        "--spark", action="store_true", default=False, help="If true, pyspark will be used to perform the preprocessing"
    )

    parser.add_argument(
        "--no_remap_ids",
        action="store_true",
        default=False,
        help="If true, the node ids of the input dataset will not be remapped to random integer ids",
    )

    parser.add_argument(
        "--sequential_train_nodes",
        action="store_true",
        default=False,
        help="If true, the train nodes will be given ids 0 to num train nodes",
    )

    parser.add_argument(
        "--src_column",
        metavar="src_column",
        required=False,
        type=int,
        default=None,
        help="The column id of the src column",
    )

    parser.add_argument(
        "--num_nodes",
        metavar="num_nodes",
        required = False,
        type = int,
        default = None,
        help = "The number of nodes must be specified if no_remap_ids"
    )

    parser.add_argument(
        "--dst_column",
        metavar="dst_column",
        required=False,
        type=int,
        default=None,
        help="The column id of the dst column",
    )

    parser.add_argument(
        "--edge_type_column",
        metavar="edge_type_column",
        required=False,
        type=int,
        default=None,
        help="The column id which denotes the edge weight column",
    )

    parser.add_argument(
        "--edge_weight_column",
        metavar="edge_weight_column",
        required=False,
        type=int,
        default=None,
        help="The column id which denotes the edge weight column",
    )

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()
    if args.dataset == "custom" and (args.src_column is None or args.dst_column is None):
        parser.error("When using a custom dataset, src column and dst column must be specified")

    if args.output_directory == "":
        args.output_directory = args.dataset

    if args.overwrite and Path(args.output_directory).exists():
        shutil.rmtree(args.output_directory)

    dataset_dict = {
        "FB15K": fb15k.FB15K,
        "FB15K_237": fb15k_237.FB15K237,
        "LIVEJOURNAL": livejournal.Livejournal,
        "TWITTER": twitter.Twitter,
        "FREEBASE86M": freebase86m.Freebase86m,
        "OGBL_WIKIKG2": ogbl_wikikg2.OGBLWikiKG2,
        "OGBL_CITATION2": ogbl_citation2.OGBLCitation2,
        "OGBL_PPA": ogbl_ppa.OGBLPpa,
        "OGBN_ARXIV": ogbn_arxiv.OGBNArxiv,
        "OGBN_PRODUCTS": ogbn_products.OGBNProducts,
        "OGBN_PAPERS100M": ogbn_papers100m.OGBNPapers100M,
        "OGB_WIKIKG90MV2": ogb_wikikg90mv2.OGBWikiKG90Mv2,
        "OGB_MAG240M": ogb_mag240m.OGBMag240M,
        "OGBL_COLLAB": ogbl_collab.OGBLCollab,
    }

    dataset = dataset_dict.get(args.dataset.upper())
    if dataset is not None:
        print("Using existing dataset of", args.dataset.upper())
        dataset = dataset(args.output_directory, spark=args.spark)
        dataset.download(args.overwrite)
        dataset.preprocess(
            num_partitions=args.num_partitions,
            remap_ids=not args.no_remap_ids,
            splits=args.dataset_split,
            sequential_train_nodes=args.sequential_train_nodes,
            partitioned_eval=args.partitioned_eval,
        )

    else:
        print("Preprocess custom dataset")

        # custom link prediction dataset
        dataset = custom.CustomLinkPredictionDataset(
            output_directory=args.output_directory,
            files=args.edges,
            delim=args.delim,
            dataset_name=args.dataset,
            spark=args.spark,
        )
        dataset.preprocess(
            num_partitions=args.num_partitions,
            remap_ids=not args.no_remap_ids,
            splits=args.dataset_split,
            partitioned_eval=args.partitioned_eval,
            sequential_train_nodes=args.sequential_train_nodes,
            src_column=args.src_column,
            num_nodes = args.num_nodes,
            dst_column=args.dst_column,
            edge_type_column=args.edge_type_column,
            edge_weight_column=args.edge_weight_column,
        )


if __name__ == "__main__":
    main()
