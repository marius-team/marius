from pathlib import Path
import numpy as np
import json


def idx_converter(dict_file, embedding_file):
    mapping = np.loadtxt(dict_file, dtype=str, delimiter="\t")
    mapping_dict = dict(mapping)

    num = len(mapping_dict)
    embeddings = np.fromfile(embedding_file, np.float32).reshape((num, -1))

    raw_emb_dict = dict()
    for rid in mapping_dict.keys():
        iid = int(mapping_dict.get(rid))
        emb = embeddings[iid]
        raw_emb_dict.update({rid: emb})

    return raw_emb_dict


def get_emb_dicts(output_dir=None):
    trained_base_dir = Path("./data/")
    temp = output_dir if output_dir != None else None
    output_dir = "./output_dir" if output_dir == None else output_dir

    node_save_file = trained_base_dir / Path("node_embedding_dict.json")
    lhs_rel_save_file = trained_base_dir / Path("lhs_rel_embedding_dict.json")
    rhs_rel_save_file = trained_base_dir / Path("rhs_rel_embedding_dict.json")
    save_files = [node_save_file, lhs_rel_save_file, rhs_rel_save_file]

    nodes_dict_file = Path(output_dir) / Path("node_mapping.txt")
    nodes_embedding_file = trained_base_dir / Path("marius/embeddings/embeddings.bin")
    node_embs = idx_converter(nodes_dict_file, nodes_embedding_file)

    rels_dict_file = Path(output_dir) / Path("rel_mapping.txt")
    lhs_rels_embedding_file = trained_base_dir / Path("marius/relations/src_relations.bin")
    lhs_rel_embs = idx_converter(rels_dict_file, lhs_rels_embedding_file)

    rhs_rels_embedding_file = trained_base_dir / Path("marius/relations/dst_relations.bin")
    rhs_rel_embs = idx_converter(rels_dict_file, rhs_rels_embedding_file)

    emb_dicts = [node_embs, lhs_rel_embs, rhs_rel_embs]

    return emb_dicts


def main():
    emb_dicts = convert_to_tsv("./output_dir")


if __name__ == '__main__':
    main()
