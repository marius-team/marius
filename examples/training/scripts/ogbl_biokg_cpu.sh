# preprocess the ogbl_biokg graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbl_biokg

# run marius on the preprocessed input
marius_train examples/training/configs/ogbl_biokg_cpu.ini