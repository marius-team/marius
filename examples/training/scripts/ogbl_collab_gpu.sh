# preprocess the ogbl_collab graph and put preprocessed graph into output dir
marius_preprocess ogbl_collab output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_collab_gpu.ini info