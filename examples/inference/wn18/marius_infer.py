import torch
import numpy as np
import pandas as pd
from pathlib import Path
from marius.tools import postprocess


trained_base_dir = Path("../../../data/")
output_dir = Path("../../../output_dir")

node_id_dict_file = output_dir / Path("node_mapping.txt")
rel_id_dict_file = output_dir / Path("rel_mapping.txt")

node_emb_file = trained_base_dir / Path("marius/embeddings/embeddings.bin")
rel_emb_file = trained_base_dir / Path("marius/relations/src_relations.bin")

# definition files for nodes ids in raw data
def_file = output_dir / Path("wordnet-mlj12-definitions.txt")


# return tensor form embeddings and embeddings dict
def tensor_from_file(choice, is_src_rel=True):
    if choice == "node":
        mapping = dict(np.loadtxt(node_id_dict_file, dtype=str, delimiter='\t'))
        embs = postprocess.get_embs(node_id_dict_file, node_emb_file)
    else:
        mapping = dict(np.loadtxt(rel_id_dict_file, dtype=str, delimiter='\t'))
        embs = postprocess.get_embs(rel_id_dict_file, rel_emb_file)

    emb_dict = []
    for k in mapping.keys():
        emb_dict.append([k, embs[int(mapping.get(k))]])
    emb_dict = dict(emb_dict)

    return torch.tensor(embs), emb_dict


# return embedding
def lookup_embedding(choice, id, emb_dict):
    if choice == "node":
        id = def_dict.get(id)

    return emb_dict.get(str(id))


# read in definition from raw data
def read_defs(def_file):
    node_names = np.array([])
    node_ids = []
    for chunk in pd.read_csv(def_file, sep="\t", header=0, chunksize=5**10, usecols=[0,1], dtype = str):
        node_ids = np.append(node_ids, chunk[chunk.columns.values[0]])
        node_names = np.append(node_names, chunk[chunk.columns.values[1]])

    return dict(zip(node_names, node_ids)), dict(zip(node_ids, node_names))


# return in raw dicts
def raw_internal_converter(mapping_file):
    mapping = np.loadtxt(mapping_file, delimiter='\t', dtype=str)
    key = mapping[:,1]
    val = mapping[:,0]
    
    return dict(zip(key, val))


def compute_contextual_embedding(src_emb, rel_emb):
    return src_emb * rel_emb


def infer_topk_nodes(k, src_emb, rel_emb, node_embeddings):
    contextual_embedding = compute_contextual_embedding(src_emb, rel_emb)
    all_scores = torch.matmul(node_embeddings, torch.tensor(contextual_embedding))
    val, idx = all_scores.topk(k, largest = True)
    top_idx = [node_in_raw_dict.get(str(i.item())) for i in idx]
    top_k_nodes = [inv_def_dict.get(i) for i in top_idx]

    return val, top_k_nodes


# return node id of top k scores
def topk_nodes(k, all_scores):
    val, idx = all_scores.topk(k, largest = True)
    return val,idx


# GLOBAL VARIABLES generate necessary dicts
node_in_raw_dict = raw_internal_converter(node_id_dict_file)
rel_in_raw_dict = raw_internal_converter(rel_id_dict_file)
def_dict, inv_def_dict = read_defs(def_file)


def main():
    # This part of the notebook shows the steps of doing inferencing with Marius output for 
    # dataset wn18. The example inference we used here has a node name of __wisconsin_NN_2 and a 
    # relationtype of _instance_hypernym.

    # example("__wisconsin_NN_2", "_instance_hypernym")

    node_embeddings, node_emb_dict = tensor_from_file("node")
    relation_embeddings, rel_emb_dict = tensor_from_file("rel")

    src_node = "__wisconsin_NN_2"
    relation = "_instance_hypernym"

    src_emb = lookup_embedding("node", src_id, node_emb_dict)
    rel_emb = lookup_embedding("rel", relation, rel_emb_dict)
    
    scores, topk = infer_topk_nodes(3, src_emb, rel_emb, node_embeddings)
    print(topk)


    return(0)



if __name__ == '__main__':
    main()
