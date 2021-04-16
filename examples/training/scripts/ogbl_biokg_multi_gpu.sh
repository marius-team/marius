# preprocess the ogbl_biokg graph and put preprocessed graph into output dir
marius_preprocess ogbl_biokg output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_biokg_multi_gpu.ini info