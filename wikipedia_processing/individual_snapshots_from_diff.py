import os
import multiprocessing
import dask.dataframe as dd
import numpy as np
import subprocess
import boto3

USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "wikipedia_dataset")
COMBINED_DATA_DIR = os.path.join(DATA_DIR, "combined_graph_snapshots")
INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "individual_graph_snapshots")
UPDATE_DIR_PREFIX = "update_"
COL_NAMES = ["src_id", "edge_type", "dst_id"]
COMPRESSED_TAR_NAME = "individual_graph_non_preprocessed.tar.bz2"
BUCKET_NAME = "wikidata-update-history"

def load_dir_from_dir(dir_path):
    df_path = os.path.join(dir_path, "graph.csv")
    return dd.read_csv(df_path, header = None, names = COL_NAMES)

def process_single_directory(curr_dir_path):
    # Iterate through its directories one at a time
    print("Processing dir", curr_dir_path)
    curr_dir_name = os.path.basename(curr_dir_path)
    curr_id = int(curr_dir_name.split("_")[1].strip())

    # See if we have already processed this
    save_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, curr_dir_name)
    save_path = os.path.join(save_dir, "graph.csv")
    if os.path.exists(save_path):
        print("Already processed this directory")
        return
    os.makedirs(save_dir, exist_ok = True)
    
    # Determine the prev dir path
    if curr_id == 0:
        prev_dir_name = "initial_snapshot"
    else:
        prev_dir_name = UPDATE_DIR_PREFIX + str(curr_id - 1)
    prev_dir_path = os.path.join(COMBINED_DATA_DIR, prev_dir_name)
    print("Determined previous dir of", prev_dir_path)

    # Load both the dataframes
    curr_snapshot_df = load_dir_from_dir(curr_dir_path)
    prev_snapshot_df = load_dir_from_dir(prev_dir_path)
    
    # Find all the nodes that are in the curren snapshot but not in the prev snapshot
    merged_df = curr_snapshot_df.merge(prev_snapshot_df, on = COL_NAMES, how='left', indicator=True)
    merged_df = merged_df[merged_df['_merge'] == 'left_only']
    merged_df = merged_df.drop(columns = ['_merge'])
    
    # Save the result
    if os.path.exists(save_path):
        os.remove(save_path)
    merged_df.to_csv(save_path, single_file = True, header = False, index = False)
    print("Saved result to", save_path)

JUST_UPLOAD = True
def main():
    if not JUST_UPLOAD:
        # Determine the update directories
        for dir_name in os.listdir(COMBINED_DATA_DIR):
            if dir_name[0] == '.' or UPDATE_DIR_PREFIX not in dir_name:
                continue
            
            dir_path = os.path.join(COMBINED_DATA_DIR, dir_name)
            process_single_directory(dir_path)
    
    # Create the compressed zip
    os.chdir(DATA_DIR)
    compress_command = f'tar -cvjSf {COMPRESSED_TAR_NAME} {INDIVIDUAL_GRAPH_SNAPSHOTS}'
    print("Running command", compress_command)
    subprocess.run(compress_command, shell = True, capture_output = True)

    # Then upload the data to S3
    print("Uploading file", COMPRESSED_TAR_NAME, "to S3")
    s3_client = boto3.client('s3')
    s3_client.upload_file(COMPRESSED_TAR_NAME, BUCKET_NAME, COMPRESSED_TAR_NAME)

if __name__ == "__main__":
    main()