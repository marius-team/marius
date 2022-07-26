from pathlib import Path
from marius.tools.configuration.constants import PathConstants
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import torch
import shutil

SUPPORTED_FORMATS = ["CSV", "PARQUET", "BINARY", "BIN"]


def save_df(output_df: pd.DataFrame, output_dir: Path, name: str, fmt: str, delim: str = ",", overwrite: bool = False):
    output_path = output_dir / Path(f"{name}.{fmt.lower()}")

    if output_path.exists() and not overwrite:
        raise RuntimeError(f"{output_path} already exists. Enable overwrite mode or delete/move the file to save.")

    if fmt == "CSV":
        output_df.to_csv(output_path, sep=delim, index=False)
    elif fmt == "PARQUET":
        output_df.to_parquet(output_path)
    else:
        raise RuntimeError(f"Unimplemented format: {fmt}")

    print(f"Wrote {output_path}: shape {output_df.shape}")


def map_ndarray_to_df(mapping_file: Path, array: np.ndarray, columns=None) -> pd.DataFrame:
    if columns is None:
        columns = ["id", "embedding"]

    if mapping_file.exists():
        node_mapping = pd.read_csv(mapping_file, header=None)
        raw_id = node_mapping.iloc[:, 0]
        mapped_id = node_mapping.iloc[:, 1]
        array = array[mapped_id]
    else:
        raw_id = np.arange(array.shape[0])

    output_df = pd.DataFrame([pd.Series(raw_id), pd.Series(array.tolist())]).transpose()
    output_df.columns = columns
    return output_df


class InMemoryExporter(object):

    def __init__(self,
                 model_dir: Path,
                 fmt: str = "CSV",
                 delim: str = ",",
                 overwrite: bool = False):

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
            node_embeddings = np.fromfile(node_embedding_path, np.float32).reshape(num_nodes, -1)
            output_df = map_ndarray_to_df(node_mapping_path, node_embeddings)
            save_df(output_df, output_dir, "embeddings", self.fmt, self.delim, self.overwrite)

        encoded_nodes_path = self.model_dir / "encoded_nodes.bin"
        if encoded_nodes_path.exists():
            node_embeddings = np.fromfile(encoded_nodes_path, np.float32).reshape(num_nodes, -1)
            output_df = map_ndarray_to_df(node_mapping_path, node_embeddings)
            save_df(output_df, output_dir, "encoded_nodes", self.fmt, self.delim, self.overwrite)

    def export_rel_embeddings(self, output_dir: Path):

        model = torch.jit.load(self.model_dir / PathConstants.model_file)
        rel_mapping_path = self.model_dir / PathConstants.relation_mapping_path

        model_param_dict = dict(model.named_parameters(recurse=True))

        if "relation_embeddings" in model_param_dict.keys():
            output_df = map_ndarray_to_df(rel_mapping_path, model_param_dict["relation_embeddings"])
            save_df(output_df, output_dir, "rel_embeddings", self.fmt, self.delim, self.overwrite)

        if "inverse_relation_embeddings" in model_param_dict.keys():
            output_df = map_ndarray_to_df(rel_mapping_path, model_param_dict["inverse_relation_embeddings"])
            save_df(output_df, output_dir, "inv_rel_embeddings", self.fmt, self.delim, self.overwrite)

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
                    raise RuntimeError(f"{output_path} already exists. Enable overwrite mode or delete/move the file to save.")
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
