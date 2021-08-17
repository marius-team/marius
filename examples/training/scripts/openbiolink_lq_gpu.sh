# preprocess the openbiolink_lq graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset openbiolink_lq

# run marius on the preprocessed input
marius_train examples/training/configs/openbiolink_lq_gpu.ini