# preprocess the ckg graph and put preprocessed graph into output dir
marius_preprocess ckg output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ckg_cpu.ini info