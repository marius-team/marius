# preprocess the yago_310 graph and put preprocessed graph into output dir
marius_preprocess yago_310 output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/yago_310_gpu.ini info