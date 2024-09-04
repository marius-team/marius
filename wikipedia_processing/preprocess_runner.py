import os
import boto3
import subprocess
import numpy as np
import multiprocessing

USER_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_DIR, "wikipedia_dataset")
INDIVIDUAL_GRAPH_SNAPSHOTS = os.path.join(DATA_DIR, "remapped_individual_graph_snapshots")
UPDATE_DIR_PREFIX = "update_"
OUTPUT_DIR_NAME = "marius_formatted"
COMPRESSED_TAR_NAME = "remapped_individual_graph_preprocessed.tar.bz2"
BUCKET_NAME = "wikidata-update-history"

def preprocess_worker(dirs_to_preprocess):
    for dir_path in list(dirs_to_preprocess):
        # Determine the command to run
        dir_path = str(dir_path)
        output_path = os.path.join(dir_path, OUTPUT_DIR_NAME)
        edges_path = os.path.join(dir_path, "graph.csv")

        preprocess_command = f'marius_preprocess --edges {edges_path}  --output_directory {output_path} --delim "," --src_column 0 --edge_type_column 1 --dst_column 2 '
        preprocess_command += "--dataset_split 0.8 0.1 0.1 --no_remap_ids --num_nodes 11790930 --num_rels 974 --overwrite"
        print("Running command", preprocess_command)
        subprocess.run(preprocess_command, shell = True, capture_output = True)

JUST_UPLOAD = True
NUM_WORKERS = int(0.4 * os.cpu_count())
def main():
    if not JUST_UPLOAD:
        # Determine the directories for the current snapshot
        all_snapshots_dir = []
        for dir_name in os.listdir(INDIVIDUAL_GRAPH_SNAPSHOTS):
            if dir_name[0] == '.' or "json" in dir_name:
                continue
            
            curr_dir_path = os.path.join(INDIVIDUAL_GRAPH_SNAPSHOTS, dir_name)
            all_snapshots_dir.append(curr_dir_path)
        
        # Determine the dirs for each worker
        dirs_per_worker = np.array_split(np.array(all_snapshots_dir), NUM_WORKERS)
        all_workers = []
        for curr_worker_dirs in dirs_per_worker:
            curr_worker = multiprocessing.Process(target = preprocess_worker, args = (curr_worker_dirs, ))
            curr_worker.start()
            all_workers.append(curr_worker)
        
        [worker.join() for worker in all_workers]
    
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