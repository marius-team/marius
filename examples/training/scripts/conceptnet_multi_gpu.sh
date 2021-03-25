# preprocess the conceptnet graph and put preprocessed graph into output dir
python3 tools/preprocess.py conceptnet output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/conceptnet_multi_gpu.ini info