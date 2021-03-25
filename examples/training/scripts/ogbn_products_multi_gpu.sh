# preprocess the ogbn_products graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbn_products output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbn_products_multi_gpu.ini info