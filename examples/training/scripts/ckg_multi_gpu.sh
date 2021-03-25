# preprocess the ckg graph and put preprocessed graph into output dir
python3 tools/preprocess.py ckg output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ckg_multi_gpu.ini info