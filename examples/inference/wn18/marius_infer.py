import torch
import numpy as np
import pandas as pd

# files for node ids in raw data and node ids used by Marius
nodes_raw_id_file = "../output_dir/node_mapping.txt"
nodes_in_id_file = "../output_dir/node_mapping.bin"
nodes_embedding_file = "../training_data/marius/embeddings/embeddings.bin"

# files for relation ids in raw data and relation ids used by Marius
rels_raw_id_file = "../output_dir/rel_mapping.txt"
rels_in_id_file = "../output_dir/rel_mapping.bin"
rels_embedding_file = "../training_data/marius/relations/lhs_relations.bin"

# definition files for nodes ids in raw data
def_file = "../output_dir/wordnet-mlj12-definitions.txt"


# return tensor form embeddings
def tensor_from_file(choice):
    if choice == "node":
        raw_id_file = nodes_raw_id_file
        embedding_file = nodes_embedding_file
    else:
        raw_id_file = rels_raw_id_file
        embedding_file = rels_embedding_file

    raw_ids = []
    with open(raw_id_file) as f:
        for line in f.readlines():
            raw_ids.append(line.strip())

    num = len(raw_ids)
    embeddings = np.fromfile(embedding_file, np.float32).reshape((num, -1))
    return torch.tensor(embeddings)


# return embedding
def lookup_embedding(choice, id, embeddings):
    if choice == "node":
        rid = def_dict.get(id)
        dic = node_raw_in_dict
    else:
        rid = id
        dic = rel_raw_in_dict
    
    return embeddings[dic.get(rid)]


# read in definition from raw data
def read_defs(def_file):
    node_names = np.array([])
    node_ids = []
    for chunk in pd.read_csv(def_file, sep="\t", header=0, chunksize=5**10, usecols=[0,1], dtype = str):
        node_ids = np.append(node_ids, chunk[chunk.columns.values[0]])
        node_names = np.append(node_names, chunk[chunk.columns.values[1]])

    return dict(zip(node_names, node_ids)), dict(zip(node_ids, node_names))

# return in raw dicts
def raw_internal_converter(raw_id_file, in_id_file):
    raw_ids = []
    with open(raw_id_file) as f:
        for line in f.readlines():
            raw_ids.append(line.strip())

    in_ids = np.fromfile(in_id_file, dtype = int)

    raw_in_dict = dict(zip(raw_ids, in_ids))
    in_raw_dict = dict(zip(in_ids, raw_ids))
    
    return raw_in_dict, in_raw_dict


def compute_contextual_embedding(src_emb, rel_emb):
    return src_emb * rel_emb


def infer_topk_nodes(k, src_emb, rel_emb, node_embeddings):
    contextual_embedding = compute_contextual_embedding(src_emb, rel_emb)
    all_scores = torch.matmul(node_embeddings, contextual_embedding)
    val, idx = all_scores.topk(k, largest = True)
    top_idx = [node_in_raw_dict.get(i.item()) for i in idx]
    top_k_nodes = [inv_def_dict.get(i) for i in top_idx]

    return val, top_k_nodes


# return node id of top k scores
def topk_nodes(k, all_scores):
    val, idx = all_scores.topk(k, largest = True)
    return val,idx

# GLOBAL VARIABLES generate necessary dicts
node_raw_in_dict, node_in_raw_dict = raw_internal_converter(nodes_raw_id_file, nodes_in_id_file)
rel_raw_in_dict, rel_in_raw_dict = raw_internal_converter(rels_raw_id_file, rels_in_id_file)
def_dict, inv_def_dict = read_defs(def_file)



def main():
    # This part of the notebook shows the steps of doing inferencing with Marius output for 
    # dataset wn18. The example inference we used here has a node name of __wisconsin_NN_2 and a 
    # relationtype of _instance_hypernym.

    # example("__wisconsin_NN_2", "_instance_hypernym")
    return(0)



if __name__ == '__main__':
    main()
