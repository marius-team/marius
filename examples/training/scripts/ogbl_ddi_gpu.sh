# preprocess the ogbl_ddi graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbl_ddi

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_ddi_gpu.ini