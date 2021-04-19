# preprocess the ogbn_products graph and put preprocessed graph into output dir
marius_preprocess ogbn_products output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_products_gpu.ini info