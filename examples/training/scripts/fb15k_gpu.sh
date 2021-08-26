# preprocess the fb15k graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset fb15k

# run marius on the preprocessed input
marius_train examples/training/configs/fb15k_gpu.ini