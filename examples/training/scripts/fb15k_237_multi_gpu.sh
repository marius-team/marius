# preprocess the fb15k_237 graph and put preprocessed graph into output dir
marius_preprocess fb15k_237 output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/fb15k_237_multi_gpu.ini info