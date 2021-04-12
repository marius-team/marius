# preprocess the ogbl_ppa graph and put preprocessed graph into output dir
marius_preprocess ogbl_ppa output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_ppa_gpu.ini info