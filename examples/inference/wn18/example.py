import torch
import numpy as np
import pandas as pd

# read in definition from raw data
def read_defs(def_file):
    node_names = np.array([])
    node_ids = []
    for chunk in pd.read_csv(def_file, sep="\t", header=0, chunksize=5**10, usecols=[0,1], dtype = str):
        node_ids = np.append(node_ids, chunk[chunk.columns.values[0]])
        node_names = np.append(node_names, chunk[chunk.columns.values[1]])

    return dict(zip(node_names, node_ids)), dict(zip(node_ids, node_names))

# return raw_in_dict ,embeddings
def tensor_from_file(raw_id_file, in_id_file, embedding_file):
    raw_ids = []
    with open(raw_id_file) as f:
        for line in f.readlines():
            raw_ids.append(line.strip())

    in_ids = np.fromfile(in_id_file, dtype = int)

    raw_in_dict = dict(zip(raw_ids, in_ids))
    in_raw_dict = dict(zip(in_ids, raw_ids))

    num = len(raw_ids)
    embeddings = np.fromfile(embedding_file, np.float32).reshape((num, -1))

    return raw_in_dict, in_raw_dict, torch.tensor(embeddings)


def lookup_embedding(iid, embeddings):
    return embeddings[iid]


def compute_contextual_embedding(src_emb, rel_emb):
    return src_emb * rel_emb


# return node id of top k scores
def topk_nodes(k, all_scores):
    val, idx = all_scores.topk(k, largest = True)
    return idx

def example(src_id, rel_id):

    # Define paths to the necessary output files of marius
    nodes_raw_id_file = "./output_dir/node_mapping.txt"
    nodes_in_id_file = "./output_dir/node_mapping.bin"
    nodes_embedding_file = "./training_data/marius/embeddings/embeddings.bin"

    rels_raw_id_file = "./output_dir/rel_mapping.txt"
    rels_in_id_file = "./output_dir/rel_mapping.bin"
    rels_embedding_file = "./training_data/marius/relations/lhs_relations.bin"

    def_file = "./output_dir/wordnet-mlj12-definitions.txt"


    node_raw_in_dict, node_in_raw_dict, node_embeddings = tensor_from_file(nodes_raw_id_file, nodes_in_id_file, nodes_embedding_file)
    rel_raw_in_dict, rel_in_raw_dict, rel_embeddings = tensor_from_file(rels_raw_id_file, rels_in_id_file, rels_embedding_file)
    def_dict, inv_def_dict = read_defs(def_file)

    node_rid = def_dict.get(src_id)
    node_iid = node_raw_in_dict.get(node_rid)
    rel_iid = rel_raw_in_dict.get(rel_id)


    src_emb = lookup_embedding(node_iid, node_embeddings)
    rel_emb = lookup_embedding(rel_iid, rel_embeddings)
    contextual_embedding = compute_contextual_embedding(src_emb, rel_emb)
    all_scores = torch.matmul(node_embeddings, contextual_embedding)
    top_idx = [node_in_raw_dict.get(i.item()) for i in list(topk_nodes(10, all_scores))] # raw id
    top_10 = [inv_def_dict.get(i) for i in top_idx]


    print()


def main():
    # This part of the notebook shows the steps of doing inferencing with Marius output for 
    # dataset wn18. The example inference we used here has a node name of __wisconsin_NN_2 and a 
    # relationtype of _instance_hypernym.

    example("__wisconsin_NN_2", "_instance_hypernym")




if __name__ == '__main__':
    main()