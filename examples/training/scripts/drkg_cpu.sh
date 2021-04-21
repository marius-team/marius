# preprocess the drkg graph and put preprocessed graph into output dir
marius_preprocess drkg output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/drkg_cpu.ini info