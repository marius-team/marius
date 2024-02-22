import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from marius.tools.postprocess.in_memory_exporter import InMemoryExporter

# from marius.tools.postprocess.spark_exporter import SparkExporter


def set_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert trained embeddings to desired output format and output to specified directory.\n\n"
            "Example usage:\n"
            "marius_postprocess --model_dir foo --format csv --output_dir bar"
        ),
        prog="postprocess",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_dir", metavar="model_dir", type=str, help="Directory of the trained model")
    parser.add_argument(
        "--format",
        "-f",
        metavar="format",
        default="CSV",
        help="Format of output embeddings. Choices are [csv, parquet, binary]",
    )
    parser.add_argument("--delim", metavar="delim", default=",", help="Delimiter to use for the output CSV")
    # parser.add_argument('--spark',
    #                     action='store_true',
    #                     default=False,
    #                     help='If true, pyspark will be used to perform the postprocessing')
    parser.add_argument(
        "--output_dir",
        metavar="output_dir",
        type=str,
        default=None,
        help="Output directory, if not provided the model directory will be used.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False, help="If enabled, the output directory will be overwritten"
    )

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    fmt = args.format.upper()
    delim = args.delim
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = model_dir
    else:
        output_dir = Path(output_dir)

    exporter = InMemoryExporter(model_dir, fmt=fmt, delim=delim, overwrite=args.overwrite)
    exporter.export(output_dir)


if __name__ == "__main__":
    main()
