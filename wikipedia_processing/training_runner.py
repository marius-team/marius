import os
import boto3
import subprocess
import numpy as np
import multiprocessing 
import shutil
import yaml

SNAPSHOT_RANGE = [0, 1]
USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "wikipedia_dataset")
INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "remapped_individual_graph_snapshots")
PREPROCESS_DIR = "marius_formatted"
UPDATE_DIR_PREFIX = "update_"
BASE_TRAINING_FILE_NAME = "../wikipedia_processing/base_training.yaml"
COMPRESSED_TAR_NAME = "remapped_all_trained_individual_graph.tar.bz2"
BUCKET_NAME = "wikidata-update-history"

def train_snapshot(snapshot_directory, prev_snapshot_dir):
    # First copy over the training yaml
    snapshot_train_config_path = os.path.join(snapshot_directory, "training.yaml")
    shutil.copy(BASE_TRAINING_FILE_NAME, snapshot_train_config_path)

    # Load the training yaml
    with open(snapshot_train_config_path, 'r') as reader:
        curr_config = dict(yaml.safe_load(reader))
    
    # Update the directories
    preprocess_dir = os.path.join(snapshot_directory, PREPROCESS_DIR)
    curr_config["storage"]["dataset"]["dataset_dir"] = preprocess_dir
    if prev_snapshot_dir is not None:
        curr_config["storage"]["prev_snapshot_dir"] = os.path.join(prev_snapshot_dir, PREPROCESS_DIR, "model_0") + "/"

    # Write the yaml back 
    with open(snapshot_train_config_path, 'w+') as writer:
        yaml.dump(curr_config, writer, default_flow_style = False)
    
    # Remove any existing trained models
    subprocess.run(f'rm -rf {preprocess_dir}/model_*', shell = True, capture_output = True)

    # Perform the training
    print("Starting training for config", snapshot_train_config_path)
    result = subprocess.run(f'./marius_train {snapshot_train_config_path}', shell = True, capture_output = True, text = True)

JUST_UPLOAD = False
def main():
    if not JUST_UPLOAD:
        # Build the training script
        os.chdir("../build")
        subprocess.run("make marius_train -j", shell = True, capture_output = True)

        # First train the initial snapshot dir
        initial_snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, "initial_snapshot")
        # train_snapshot(initial_snapshot_dir, None)

        # Then train each update
        for snapshot_id in range(SNAPSHOT_RANGE[0], SNAPSHOT_RANGE[1] + 1):
            snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, UPDATE_DIR_PREFIX + str(snapshot_id))
            if snapshot_id == 0:
                prev_snapshot_dir = initial_snapshot_dir
            else:
                prev_snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, UPDATE_DIR_PREFIX + str(snapshot_id - 1))
            
            train_snapshot(snapshot_dir, prev_snapshot_dir)
    
    '''
    # Create the compressed zip
    os.chdir(DATA_DIR)
    compress_command = f'tar -cvjSf {COMPRESSED_TAR_NAME} {INDIVIDUAL_GRAPH_SNAPSHOTS}'
    print("Running command", compress_command)
    subprocess.run(compress_command, shell = True, capture_output = True)

    # Then upload the data to S3
    print("Uploading file", COMPRESSED_TAR_NAME, "to S3")
    s3_client = boto3.client('s3')
    s3_client.upload_file(COMPRESSED_TAR_NAME, BUCKET_NAME, COMPRESSED_TAR_NAME)
    '''

if __name__ == "__main__":
    main()