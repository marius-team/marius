# preprocess the kinships graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset kinships

# run marius on the preprocessed input
marius_train examples/training/configs/kinships_cpu.ini