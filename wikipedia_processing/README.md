# Wikipedia Analysis 

Initial preprocess command:
```
marius_preprocess --edges /root/wikipedia_dataset/remapped_individual_graph_snapshots/initial_snapshot/graph.csv  --output_directory /root/wikipedia_dataset/remapped_individual_graph_snapshots/initial_snapshot/marius_formatted --delim "," --src_column 0 --edge_type_column 1 --dst_column 2 --dataset_split 0.8 0.1 0.1 --no_remap_ids --num_nodes 11790930 --num_rels 974
```

If any changes are made to the python code, then we need to rerun the following command in the root dir:
```
pip3 install . --no-build-isolation
```

To build marius, first run:
```
mkdir -p build
cd build
cmake ../ -DUSE_CUDA=TRUE -DUSE_OMP=TRUE
```

Run training using a command like this:
```
rm -rf /root/wikipedia_dataset/remapped_individual_graph_snapshots/update_0/marius_formatted/model_* && make marius_train -j && ./marius_train /root/wikipedia_dataset/remapped_individual_graph_snapshots/update_0/training.yaml
```

Upload the result using:
```
cd /root/wikipedia_dataset
tar -cvjSf sample_remapped_initial_trained.tar.bz2 /root/wikipedia_dataset/remapped_individual_graph_snapshots/initial_snapshot
```