#!/usr/bin/env bash

BASE_DIR=freebase_16
NUM_PART=16

CMD="python parquet_to_hdf5.py --edge_dir_in "$BASE_DIR/output_data/freebase/train" --ent_part_count $2 --entity_dir "$BASE_DIR/output_data/freebase/entities" --relation_dir "$BASE_DIR/output_data/freebase/relations" --edge_dir_out "$BASE_DIR/output_data/freebase/freebase_train" --output_path "$BASE_DIR/output_data/freebase/freebase_metadata""
$CMD

CMD="python parquet_to_hdf5.py --edge_dir_in "$BASE_DIR/output_data/freebase/valid" --ent_part_count $2 --entity_dir None --relation_dir None --edge_dir_out "$BASE_DIR/output_data/freebase/freebase_valid" --output_path None"
$CMD

CMD="python parquet_to_hdf5.py --edge_dir_in "$BASE_DIR/output_data/freebase/test" --ent_part_count $2 --entity_dir None --relation_dir None --edge_dir_out "$BASE_DIR/output_data/freebase/freebase_test" --output_path None"
$CMD
