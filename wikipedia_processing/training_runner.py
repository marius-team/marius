import os
import boto3
import subprocess
import numpy as np
import multiprocessing 
import shutil
import yaml

SNAPSHOT_RANGE = [0, 103]
USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "all_datasets")
INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "wikipedia_dataset")
PREPROCESS_DIR = "marius_formatted"
UPDATE_DIR_PREFIX = "update_"
BASE_TRAINING_FILE_NAME = "base_training.yaml"
BUCKET_NAME = "wikidata-update-history"
BASE_MODEL_NAME = "model_0"
EMBEDDINGS_PREFIX_NAME = "remapped_embeddings_"

JUST_UPLOAD = False
def train_snapshot(s3_client, snapshot_directory, prev_snapshot_dir):
    model_dir = os.path.join(snapshot_directory, PREPROCESS_DIR, BASE_MODEL_NAME)
    if not JUST_UPLOAD and not os.path.exists(model_dir):
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
            curr_config["storage"]["prev_snapshot_dir"] = os.path.join(prev_snapshot_dir, PREPROCESS_DIR, BASE_MODEL_NAME) + "/"

        # Write the yaml back 
        with open(snapshot_train_config_path, 'w+') as writer:
            yaml.dump(curr_config, writer, default_flow_style = False)

        # Perform the training
        print("Starting training for config", snapshot_train_config_path)
        result = subprocess.run(f'marius_train {snapshot_train_config_path}', shell = True, capture_output = True, text = True)
    
        # Upload the embeddings
        embeddings_path = os.path.join(snapshot_directory, PREPROCESS_DIR, BASE_MODEL_NAME, "embeddings.bin")
        save_name = EMBEDDINGS_PREFIX_NAME + os.path.basename(snapshot_directory) + ".bin"
        print("Uploading embeddings", embeddings_path, "as", save_name)
        s3_client.upload_file(embeddings_path, BUCKET_NAME, save_name)

def main():
    # First train the initial snapshot dir
    initial_snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, "initial_snapshot")
    s3_client = boto3.client('s3')
    train_snapshot(s3_client, initial_snapshot_dir, None)

    # Then train each update
    for snapshot_id in range(SNAPSHOT_RANGE[0], SNAPSHOT_RANGE[1] + 1):
        snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, UPDATE_DIR_PREFIX + str(snapshot_id))
        if snapshot_id == 0:
            prev_snapshot_dir = initial_snapshot_dir
        else:
            prev_snapshot_dir = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, UPDATE_DIR_PREFIX + str(snapshot_id - 1))
        
        train_snapshot(s3_client, snapshot_dir, prev_snapshot_dir)

if __name__ == "__main__":
    main()