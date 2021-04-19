# preprocess the conceptnet graph and put preprocessed graph into output dir
marius_preprocess conceptnet output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/conceptnet_gpu.ini info