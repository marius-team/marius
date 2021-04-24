# preprocess the hetionet graph and put preprocessed graph into output dir
marius_preprocess hetionet output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/hetionet_gpu.ini info