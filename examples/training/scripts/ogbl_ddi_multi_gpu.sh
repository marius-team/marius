# preprocess the ogbl_ddi graph and put preprocessed graph into output dir
marius_preprocess ogbl_ddi output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_ddi_multi_gpu.ini info