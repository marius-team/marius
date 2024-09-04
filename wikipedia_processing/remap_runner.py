import os
import multiprocessing
import pandas as pd
import numpy as np
import subprocess
import boto3
import json

USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "wikipedia_dataset")
NON_REMAPPED_COMBINED_DIR = os.path.join(DATA_DIR, "combined_graph_snapshots")
NON_REMAPPED_INDIVIDUAL_DIR = os.path.join(DATA_DIR, "individual_graph_snapshots")
REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "remapped_individual_graph_snapshots")
NODE_REMAP_FILE, REL_REMAP_FILE = "node_remaps.json", "rel_remaps.json"
UPDATE_DIR_PREFIX = "update_"
COL_NAMES = ["src_id", "edge_type", "dst_id"]
COMPRESSED_TAR_NAME = "remapped_individual_graph_non_preprocessed.tar.bz2"
BUCKET_NAME = "wikidata-update-history"

def get_last_dir():
    largest_id, largest_path = None, None
    for file_name in os.listdir(NON_REMAPPED_COMBINED_DIR):
        if file_name[0] == '.' or UPDATE_DIR_PREFIX not in file_name:
            continue
        
        update_id = int(file_name.split("_")[1].strip())
        if largest_id is None or update_id > largest_id:
            largest_id = update_id
            largest_path = os.path.join(NON_REMAPPED_COMBINED_DIR, file_name)
    
    return largest_path

REGENERATE_MAPPING = False
def load_node_remap(last_dir_path):
    # See if we already have the map
    node_remap_path = os.path.join(REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS, NODE_REMAP_FILE)
    if not REGENERATE_MAPPING and os.path.exists(node_remap_path):
        with open(node_remap_path, 'r') as reader:
            node_mappings = json.load(reader)
        return dict(node_mappings)

    # If not read the graph and get all the unique nodes
    graph_df = pd.read_csv(os.path.join(last_dir_path, "graph.csv"), header = None, names = COL_NAMES)
    all_unique_nodes = set(graph_df["src_id"].unique()).union(set(graph_df["dst_id"].unique()))
    node_mappings = {str(node_id) : int(idx) for idx, node_id in enumerate(all_unique_nodes)}
    
    # Write the node mappings to disk
    with open(node_remap_path, 'w+') as writer:
        json.dump(node_mappings, writer, indent = 4, sort_keys = True)

    return node_mappings

def load_rel_remap(last_dir_path):
    # See if we already have the map
    rel_remap_path = os.path.join(REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS, REL_REMAP_FILE)
    if not REGENERATE_MAPPING and os.path.exists(rel_remap_path):
        with open(rel_remap_path, 'r') as reader:
            rel_mappings = json.load(reader)
        return dict(rel_mappings)

    # If not read the graph and get all the unique nodes
    graph_df = pd.read_csv(os.path.join(last_dir_path, "graph.csv"), header = None, names = COL_NAMES)
    all_unique_rels = set(graph_df["edge_type"].unique())
    rel_mappings = {str(node_id) : int(idx) for idx, node_id in enumerate(all_unique_rels)}
    
    # Write the node mappings to disk
    with open(rel_remap_path, 'w+') as writer:
        json.dump(rel_mappings, writer, indent = 4, sort_keys = True)
    return rel_mappings

def load_dir_from_dir(dir_path):
    df_path = os.path.join(dir_path, "graph.csv")
    return pd.read_csv(df_path, header = None, names = COL_NAMES)

def remap_worker(dirs_to_process, node_remap, rel_remap):
    # Determine the save path
    for curr_dir_path in dirs_to_process:
        print("Processing dir", curr_dir_path)
        curr_dir_name = os.path.basename(curr_dir_path)
        save_dir = os.path.join(REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS, curr_dir_name)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, "graph.csv")

        # Load the dataframe 
        curr_snapshot_df = load_dir_from_dir(curr_dir_path)
        original_size = len(curr_snapshot_df.index)

        # Apply the map on the src id
        curr_snapshot_df["src_id"] = curr_snapshot_df["src_id"].astype('str').map(node_remap)
        curr_snapshot_df = curr_snapshot_df[curr_snapshot_df["src_id"] < len(node_remap)]

        # Apply the map on the dst id
        curr_snapshot_df["dst_id"] = curr_snapshot_df["dst_id"].astype('str').map(node_remap)
        curr_snapshot_df = curr_snapshot_df[curr_snapshot_df["dst_id"] < len(node_remap)]

        # Apply the map on the relationship
        curr_snapshot_df["edge_type"] = curr_snapshot_df["edge_type"].astype('str').map(rel_remap)
        curr_snapshot_df = curr_snapshot_df[curr_snapshot_df["edge_type"] < len(rel_remap)]

        # Save the result
        curr_snapshot_df = curr_snapshot_df.dropna()
        curr_snapshot_df = curr_snapshot_df.astype('int')
        filtered_size = len(curr_snapshot_df.index)
        print("Kept", round((100.0 * filtered_size)/original_size), "percent of rows")

        curr_snapshot_df.to_csv(save_path, index = False, header = False)
        print("Saved result to", save_path)

SNAPSHOTS_TO_PROCESS = -1
JUST_UPLOAD = True
NUM_WORKERS = int(0.4 * os.cpu_count())
def main():
    # Determine the snapshots to process
    if not JUST_UPLOAD:
        # Read in the node and relationship map
        os.makedirs(REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS, exist_ok = True)
        last_dir_path = get_last_dir()
        node_remap = load_node_remap(last_dir_path)
        rel_remap = load_rel_remap(last_dir_path)
        print("Have a total of", len(node_remap), "nodes and", len(rel_remap), "relationships")

        # Determine the update directories
        all_remap_dirs = []
        for dir_name in os.listdir(NON_REMAPPED_INDIVIDUAL_DIR):
            if dir_name[0] == '.':
                continue
            
            # Determine the update it
            dir_path = os.path.join(NON_REMAPPED_INDIVIDUAL_DIR, dir_name)
            if "initial" in dir_name:
                all_remap_dirs.append(dir_path)
            else:
                update_id = int(dir_name.split("_")[1].strip())
                if SNAPSHOTS_TO_PROCESS <= 0 or update_id <= SNAPSHOTS_TO_PROCESS:
                    all_remap_dirs.append(dir_path)
        
        # Determine the dirs for each worker
        dirs_per_worker = np.array_split(np.array(all_remap_dirs), NUM_WORKERS)
        all_workers = []
        for curr_worker_dirs in dirs_per_worker:
            curr_worker = multiprocessing.Process(target = remap_worker, args = (curr_worker_dirs, node_remap, rel_remap))
            curr_worker.start()
            all_workers.append(curr_worker)
        
        [worker.join() for worker in all_workers]
    
    # Create the compressed zip
    os.chdir(DATA_DIR)
    compress_command = f'tar -cvjSf {COMPRESSED_TAR_NAME} {REMAPPED_INDIVIDUAL_GRAPH_SNAPSHOTS}'
    print("Running command", compress_command)
    subprocess.run(compress_command, shell = True, capture_output = True)

    # Then upload the data to S3
    print("Uploading file", COMPRESSED_TAR_NAME, "to S3")
    s3_client = boto3.client('s3')
    s3_client.upload_file(COMPRESSED_TAR_NAME, BUCKET_NAME, COMPRESSED_TAR_NAME)

if __name__ == "__main__":
    main()