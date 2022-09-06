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
        "--columns",
        metavar="columns",
        nargs="*",
        required=False,
        type=int,
        default=[0, 1, 2],
        help="List of column ids of input delimited files which denote the src node, edge-type, and dst node of edges.",
    )

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()

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
    }

    dataset = dataset_dict.get(args.dataset.upper())
    if dataset is not None:
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
            columns=args.columns,
        )


if __name__ == "__main__":
    main()
