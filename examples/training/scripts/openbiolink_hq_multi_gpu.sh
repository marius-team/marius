# preprocess the openbiolink_hq graph and put preprocessed graph into output dir
marius_preprocess openbiolink_hq output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/openbiolink_hq_multi_gpu.ini info