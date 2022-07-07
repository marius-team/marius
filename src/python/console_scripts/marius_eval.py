import argparse
import sys

import marius as m


def main():
    parser = argparse.ArgumentParser(description="Configuration file based evaluation", prog="eval")

    parser.add_argument(
        "config",
        metavar="config",
        type=str,
        help=(
            "Path to YAML configuration file that describes the evaluation process. See documentation"
            " docs/config_interface for more details."
        ),
    )

    args = parser.parse_args()
    config = m.config.loadConfig(args.config, save=True)
    m.manager.marius_eval(config)


if __name__ == "__main__":
    sys.exit(main())
