from typing import Optional

import fire
import json
import pyarrow.parquet as pq
import h5py
import os
from pathlib import Path
import pandas

def generate_edge_path_file(
        edge_dir_in: str,
        ent_part_count: int,
        edge_dir_out: str,
        entity_dir: Optional[str],
        relation_dir: Optional[str],
        output_path: Optional[str]
) -> None:
    Path(edge_dir_out).mkdir(parents=True, exist_ok=True)

    FORMAT_VERSION_ATTR = "format_version"
    FORMAT_VERSION = 1

    for lhs_p in range(0, ent_part_count):
        for rhs_p in range(0, ent_part_count):
            f1 = h5py.File(f"{edge_dir_out}/edges_{lhs_p}_{rhs_p}.h5", "w")
            edge = pq.read_table(f"{edge_dir_in}/part_left={lhs_p}/part_right={rhs_p}")
            data = edge.to_pandas().to_numpy()
            f1.create_dataset("lhs", (data.shape[0],), dtype='i', data=data[:, 0])
            f1.create_dataset("rhs", (data.shape[0],), dtype='i', data=data[:, 1])
            f1.create_dataset("rel", (data.shape[0],), dtype='i', data=data[:, 2])
            f1.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
            f1.close()
            print(f"edges_{lhs_p}_{rhs_p}  done")

    if output_path is None:
        return

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if entity_dir is not None:
        # TODO change to entity type
        entity_name = 'all'
        for partition in range(0, ent_part_count):
            df_entities = pq.read_table(f"{entity_dir}/partition={partition}").to_pandas()
            ent_list = df_entities['ent'].values.tolist()
            json.dump(ent_list, open(f"{output_path}/entity_names_{entity_name}_{partition}.json", "w"))

            with open(f"{output_path}/entity_count_{entity_name}_{partition}.txt", "w") as tf:
                tf.write(f"{str(df_entities.count()['ent'])}\n")

            print(f"entity partition {partition} done")

    if relation_dir is not None:
        df_relations = pq.read_table(relation_dir).to_pandas()
        rel_list = df_relations['relation'].values.tolist()
        json.dump(rel_list, open(f"{output_path}/dynamic_rel_names.json", "w"))

        with open(f"{output_path}/dynamic_rel_count.txt", "w") as tf:
            tf.write(f"{str(df_relations.count()['relation'])}\n")

        print(f"relations done")


if __name__ == "__main__":
    fire.Fire(generate_edge_path_file)