import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants

import torch  # isort:skip

SUPPORTED_FORMATS = ["CSV", "PARQUET", "BINARY", "BIN"]


def get_ordered_raw_ids(mapping_path):
    assert mapping_path.exists()

    mapping = pd.read_csv(mapping_path, header=None)
    raw_id = mapping.iloc[:, 0]
    mapped_id = mapping.iloc[:, 1]

    sorted_args = np.argsort(mapped_id)
    raw_id = raw_id[sorted_args]

    return raw_id


def save_df(output_df: pd.DataFrame, output_dir: Path, name: str, fmt: str, delim: str = ",", overwrite: bool = False):
    output_path = output_dir / Path(f"{name}.{fmt.lower()}")

    if output_path.exists() and not overwrite:
        raise RuntimeError(f"{output_path} already exists. Enable overwrite mode or delete/move the file to save.")

    if fmt == "CSV":
        with np.printoptions(linewidth=10000):
            output_df.to_csv(output_path, sep=delim, index=False, encoding="utf8")
    elif fmt == "PARQUET":
        output_df.to_parquet(output_path)
    else:
        raise RuntimeError(f"Unimplemented format: {fmt}")

    print(f"Wrote {output_path}: shape {output_df.shape}")


class InMemoryExporter(object):
    def __init__(self, model_dir: Path, fmt: str = "CSV", delim: str = ",", overwrite: bool = False):
        fmt = fmt.upper()

        if not model_dir.exists():
            raise RuntimeError(f"Model directory not found {model_dir}")

        if fmt not in SUPPORTED_FORMATS:
            raise RuntimeError(f"Unsupported format {fmt}, must be one of {SUPPORTED_FORMATS}")

        self.model_dir = model_dir
        self.fmt = fmt
        self.delim = delim
        self.overwrite = overwrite
        self.config = OmegaConf.load(model_dir / PathConstants.saved_full_config_file_name)

    def export_node_embeddings(self, output_dir: Path):
        num_nodes = self.config.storage.dataset.num_nodes
        node_embedding_path = self.model_dir / "embeddings.bin"
        node_mapping_path = self.model_dir / PathConstants.node_mapping_file

        if node_embedding_path.exists():
            raw_id = get_ordered_raw_ids(node_mapping_path)
        else:
            raw_id = np.arange(num_nodes)

        if node_embedding_path.exists():
            save_df(
                pd.DataFrame(
                    np.array(
                        [raw_id, list(np.fromfile(node_embedding_path, np.float32).reshape(num_nodes, -1))],
                        dtype=object,
                    ).T,
                    columns=["id", "embedding"],
                ),
                output_dir,
                "embeddings",
                self.fmt,
                self.delim,
                self.overwrite,
            )

        encoded_nodes_path = self.model_dir / "encoded_nodes.bin"
        if encoded_nodes_path.exists():
            save_df(
                pd.DataFrame(
                    np.array(
                        [raw_id, list(np.fromfile(encoded_nodes_path, np.float32).reshape(num_nodes, -1))], dtype=object
                    ).T,
                    columns=["id", "embedding"],
                ),
                output_dir,
                "encoded_nodes",
                self.fmt,
                self.delim,
                self.overwrite,
            )

    def export_rel_embeddings(self, output_dir: Path):
        num_rels = self.config.storage.dataset.num_relations
        model = torch.jit.load(self.model_dir / PathConstants.model_file).to("cpu")
        rel_mapping_path = self.model_dir / PathConstants.relation_mapping_path

        if rel_mapping_path.exists():
            raw_id = get_ordered_raw_ids(rel_mapping_path)
        else:
            raw_id = np.arange(num_rels)

        model_param_dict = dict(model.named_parameters(recurse=True))

        if "relation_embeddings" in model_param_dict.keys():
            save_df(
                pd.DataFrame(
                    np.array([raw_id, list(model_param_dict["relation_embeddings"].detach().numpy())], dtype=object).T,
                    columns=["id", "embedding"],
                ),
                output_dir,
                "relation_embeddings",
                self.fmt,
                self.delim,
                self.overwrite,
            )

        if "inverse_relation_embeddings" in model_param_dict.keys():
            save_df(
                pd.DataFrame(
                    np.array(
                        [raw_id, list(model_param_dict["inverse_relation_embeddings"].detach().numpy())], dtype=object
                    ).T,
                    columns=["id", "embedding"],
                ),
                output_dir,
                "inverse_relation_embeddings",
                self.fmt,
                self.delim,
                self.overwrite,
            )

    def export_model(self, output_dir: Path):
        model_path = self.model_dir / PathConstants.model_file
        output_path = Path(f"{output_dir}/model.pt")

        if model_path != output_path:
            if output_dir.__str__().startswith("s3://"):
                import s3fs

                s3 = s3fs.S3FileSystem()
                s3.put(model_path, output_path)
            else:
                if output_path.exists() and not self.overwrite:
                    raise RuntimeError(
                        f"{output_path} already exists. Enable overwrite mode or delete/move the file to save."
                    )
                shutil.copy(model_path, output_path)
                print(f"Wrote {output_path}")

    def copy_model(self, output_dir: Path):
        if self.model_dir != output_dir:
            if output_dir.__str__().startswith("s3://"):
                import s3fs

                s3 = s3fs.S3FileSystem()
                s3.put(self.model_dir, output_dir)
            else:
                shutil.copytree(self.model_dir, output_dir, dirs_exist_ok=self.overwrite)

    def export(self, output_dir: Path):
        if self.fmt.startswith("BIN"):
            self.copy_model(output_dir)
        else:
            if not output_dir.__str__().startswith("s3://"):
                output_dir.mkdir(parents=True, exist_ok=True)
            self.export_node_embeddings(output_dir)
            self.export_rel_embeddings(output_dir)
            self.export_model(output_dir)
