# preprocess the openbiolink_lq graph and put preprocessed graph into output dir
python3 tools/preprocess.py openbiolink_lq output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/openbiolink_lq_multi_gpu.ini info