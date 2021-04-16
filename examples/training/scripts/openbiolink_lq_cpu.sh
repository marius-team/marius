# preprocess the openbiolink_lq graph and put preprocessed graph into output dir
marius_preprocess openbiolink_lq output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/openbiolink_lq_cpu.ini info