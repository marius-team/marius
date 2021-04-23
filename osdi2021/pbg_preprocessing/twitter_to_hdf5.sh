#!/usr/bin/env bash

BASE_DIR=twitter_16
NUM_PART=16

CMD="python parquet_to_hdf5.py --edge_dir_in "freebase16/output_data/twitter/train" --ent_part_count $2 --entity_dir "$BASE_DIR/output_data/twitter/entities" --relation_dir "$BASE_DIR/output_data/twitter/relations" --edge_dir_out "$BASE_DIR/output_data/twitter/twitter_train" --output_path "$BASE_DIR/output_data/twitter/twitter_metadata""
$CMD

CMD="python parquet_to_hdf5.py --edge_dir_in "$BASE_DIR/output_data/twitter/valid" --ent_part_count $2 --entity_dir None --relation_dir None --edge_dir_out "$BASE_DIR/output_data/twitter/twitter_valid" --output_path None"
$CMD

CMD="python parquet_to_hdf5.py --edge_dir_in "$BASE_DIR/output_data/twitter/test" --ent_part_count $2 --entity_dir None --relation_dir None --edge_dir_out "$BASE_DIR/output_data/twitter/twitter_test" --output_path None"
$CMD
