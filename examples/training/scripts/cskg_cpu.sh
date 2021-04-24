# preprocess the cskg graph and put preprocessed graph into output dir
marius_preprocess cskg output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/cskg_cpu.ini info