# preprocess the fb15k graph and put preprocessed graph into output dir
marius_preprocess fb15k output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/fb15k_multi_gpu.ini info