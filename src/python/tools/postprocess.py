from pathlib import Path

import numpy as np


def idx_converter(raw_id_file, in_id_file, embedding_file):
    raw_ids = []
    with open(raw_id_file) as f:
        for line in f.readlines():
            raw_ids.append(line.strip())

    in_ids = np.fromfile(in_id_file, dtype=int)
    raw_in_dict = dict(zip(raw_ids, in_ids))

    num = len(raw_ids)
    embeddings = np.fromfile(embedding_file, np.float32).reshape((num, -1))

    raw_emb_dict = []
    for rid in raw_ids:
        iid = raw_in_dict.get(rid)
        emb = embeddings[iid]
        raw_emb_dict.append(np.append(rid, emb))

    return raw_emb_dict


def convert_to_tsv(output_dir=None):
    temp = output_dir if output_dir != None else None
    output_dir = "./output_dir" if output_dir == None else output_dir

    nodes_raw_id_file = Path(output_dir) / Path("node_mapping.txt")
    nodes_in_id_file = Path(output_dir) / Path("node_mapping.bin")
    nodes_embedding_file = Path("./training_data/marius/embeddings/embeddings.bin")
    node_embs = np.array(idx_converter(nodes_raw_id_file, nodes_in_id_file, nodes_embedding_file))

    rels_raw_id_file = Path(output_dir) / Path("rel_mapping.txt")
    rels_in_id_file = Path(output_dir) / Path("rel_mapping.bin")
    lhs_rels_embedding_file = Path("./training_data/marius/relations/lhs_relations.bin")
    lhs_rel_embs = np.array(idx_converter(rels_raw_id_file, rels_in_id_file, lhs_rels_embedding_file))

    rhs_rels_embedding_file = Path("./training_data/marius/relations/rhs_relations.bin")
    rhs_rel_embs = np.array(idx_converter(rels_raw_id_file, rels_in_id_file, rhs_rels_embedding_file))

    if temp != None:
        np.savetxt((Path("./training_data/node_embedding.tsv")), node_embs, fmt="%f", delimiter='\t', newline='\n')
        np.savetxt((Path("./training_data/edge_lhs_embedding.tsv")), lhs_rel_embs, fmt="%f", delimiter='\t',
                   newline='\n')
        np.savetxt((Path("./training_data/edge_rhs_embedding.tsv")), rhs_rel_embs, fmt="%f", delimiter='\t',
                   newline='\n')

        return node_embs, lhs_rel_embs, rhs_rel_embs
    else:
        return node_embs, lhs_rel_embs, rhs_rel_embs


def main():
    n_emb, lhs_emb, rhs_emb = convert_to_tsv()


if __name__ == '__main__':
    main()
