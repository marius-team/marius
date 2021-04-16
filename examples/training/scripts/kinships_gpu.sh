# preprocess the kinships graph and put preprocessed graph into output dir
marius_preprocess kinships output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/kinships_gpu.ini info