# preprocess the hetionet graph and put preprocessed graph into output dir
python3 tools/preprocess.py hetionet output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/hetionet_multi_gpu.ini info