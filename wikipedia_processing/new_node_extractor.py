import pandas as pd
import os
import boto3
import numpy as np
import time

USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "all_datasets")
INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "wikipedia_dataset")
UPDATE_DIR_PREFIX = "update_"
BUCKET_NAME = "wikidata-update-history"
NEW_NODES_OBJECT_PREFIX = "remapped_new_nodes_"

SNAPSHOT_RANGE = [0, 103] # [0, 104]
def new_node_creator():
    s3_client = boto3.client('s3')
    nodes_already_seem = pd.Series()
    for snapshot_id in range(SNAPSHOT_RANGE[0], SNAPSHOT_RANGE[1] + 1):
        if snapshot_id == 0:
            snapshot_name = "initial_snapshot"
        else:
            snapshot_name = "update_" + str(snapshot_id - 1)
        
        snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, snapshot_name)
        save_path = os.path.join(snapshot_dir, "new_nodes.npy")
        
        # Load in the graph
        print("Processing dir", snapshot_dir)
        graph_path = os.path.join(snapshot_dir, "graph.csv")
        graph_df = pd.read_csv(graph_path, header = None, names = ["src_node", "edge_type", "dst_node"])
        
        # Get the unique nodes in the graph
        all_nodes = pd.concat([graph_df["src_node"], graph_df["dst_node"]]).drop_duplicates()
        new_nodes = all_nodes[~all_nodes.isin(nodes_already_seem)].values

        # Write those to disk and upload to S3
        np.save(save_path, new_nodes)
        upload_name = NEW_NODES_OBJECT_PREFIX + snapshot_name + ".npy"
        print("Uploading file", upload_name, "with", new_nodes.shape[0], "nodes")
        s3_client.upload_file(save_path, BUCKET_NAME, upload_name)

        # Update the nodes already seem
        nodes_already_seem = pd.concat([all_nodes, nodes_already_seem]).drop_duplicates()

NEW_NODE_EMBEDDINGS_PREFIX = "remapped_new_node_embeddings_"
SLEEP_TIME = 60 * 10
def delta_creator():
    s3_client = boto3.client('s3')
    found_new_results = True
    while found_new_results:
        found_new_results = False
        for snapshot_name in os.listdir(INDIVIDUAL_GRAPH_SNAPSHOTS):
            if snapshot_name[0] == '.' or "json" in snapshot_name:
                continue
            
            # Get the paths and make sure they exists
            snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, snapshot_name)
            model_dir = os.path.join(snapshot_dir, "marius_formatted", "model_0")
            new_nodes_path = os.path.join(snapshot_dir, "new_nodes.npy")
            embeddings_path = os.path.join(model_dir, "embeddings.bin")
            if not os.path.exists(new_nodes_path) or not os.path.exists(embeddings_path):
                continue
            
            # Determine the save path and see if we already processed this one
            save_path = os.path.join(snapshot_dir, "new_node_embeddings.npy")
            if os.path.exists(save_path):
                continue

            # Read in the new nodes and embeddings
            print("Processing snapshot", snapshot_name)
            new_nodes = np.load(new_nodes_path)
            node_embeddings = np.fromfile(embeddings_path, dtype = np.float32).reshape(-1, 128)
            
            # Save the new node embeddings and upload to S3
            new_node_embeddings = node_embeddings[new_nodes]
            np.save(save_path, new_node_embeddings)
            upload_name = NEW_NODE_EMBEDDINGS_PREFIX + snapshot_name + ".npy"
            print("Uploading result to", upload_name)
            s3_client.upload_file(save_path, BUCKET_NAME, upload_name)

            found_new_results = True

if __name__ == "__main__":
    # new_node_creator()
    delta_creator()