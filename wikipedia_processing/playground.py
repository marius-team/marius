import os
import multiprocessing
import dask.dataframe as dd
import pandas as pd
import numpy as np
import subprocess
import boto3
import torch

COL_NAMES = ["src_id", "edge_type", "dst_id"]
def load_dir_from_dir(dir_path):
    df_path = os.path.join(dir_path, "graph.csv")
    return dd.read_csv(df_path, header = None, names = COL_NAMES)

def similar_comparsion():
    initial_snapshot_df = load_dir_from_dir("/root/wikipedia_dataset/remapped_individual_graph_snapshots/initial_snapshot")
    first_update_df = load_dir_from_dir("/root/wikipedia_dataset/remapped_individual_graph_snapshots/update_0")
    
    # Get the unique nodes in each snapshot
    inital_nodes = dd.concat([initial_snapshot_df["src_id"], initial_snapshot_df["dst_id"]]).drop_duplicates().to_frame("node_id")
    first_nodes = dd.concat([first_update_df["src_id"], first_update_df["dst_id"]]).drop_duplicates().to_frame("node_id")

    # Determine nodes only in update 0
    merged_df = first_nodes.merge(inital_nodes, on = "node_id", how='left', indicator=True)
    merged_df = merged_df[merged_df['_merge'] == 'left_only']
    print(merged_df.shape[0].compute(), first_nodes.shape[0].compute())

EMBEDDING_DIM = 128
def get_cosine_similarity():
    # Get the node in the first snapshot
    intial_df = pd.read_csv("/root/wikipedia_dataset/remapped_individual_graph_snapshots/update_0/graph.csv", header = None, names = COL_NAMES)
    inital_nodes = pd.concat([intial_df["src_id"], intial_df["dst_id"]]).drop_duplicates().values
    nodes_to_keep = torch.from_numpy(inital_nodes)

    # Load the embeddings for both of the snapshots
    initial_embeddings = np.fromfile("/root/wikipedia_dataset/remapped_individual_graph_snapshots/initial_snapshot/marius_formatted/model_0/embeddings.bin", np.float32)
    initial_embeddings = torch.from_numpy(initial_embeddings.reshape(-1, EMBEDDING_DIM))
    initial_embeddings = initial_embeddings[nodes_to_keep]
    print("Initial embeddings distribution", torch.mean(initial_embeddings), torch.std(initial_embeddings), "and having shape", initial_embeddings.shape)
    
    updated_embeddings = np.fromfile("/root/wikipedia_dataset/remapped_individual_graph_snapshots/update_0/marius_formatted/model_0/embeddings.bin", np.float32)
    updated_embeddings = torch.from_numpy(updated_embeddings.reshape(-1, EMBEDDING_DIM))
    updated_embeddings = updated_embeddings[nodes_to_keep]
    print("Updated embeddings distribution", torch.mean(updated_embeddings), torch.std(updated_embeddings), "and having shape", updated_embeddings.shape)

    cosine_similarity = torch.nn.functional.cosine_similarity(updated_embeddings, initial_embeddings)
    print("Similarity distribution", torch.mean(cosine_similarity), torch.std(cosine_similarity), "and having space", cosine_similarity.shape)

if __name__ == "__main__":
    get_cosine_similarity()