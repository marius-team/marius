# preprocess the ogbn_products graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbn_products

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_products_multi_gpu.ini