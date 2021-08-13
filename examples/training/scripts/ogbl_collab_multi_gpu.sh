# preprocess the ogbl_collab graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbl_collab

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_collab_multi_gpu.ini