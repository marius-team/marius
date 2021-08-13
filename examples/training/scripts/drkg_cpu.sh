# preprocess the drkg graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset drkg

# run marius on the preprocessed input
marius_train examples/training/configs/drkg_cpu.ini