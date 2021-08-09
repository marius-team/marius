# preprocess the ogbl_ppa graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbl_ppa

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_ppa_multi_gpu.ini