# preprocess the kinships graph and put preprocessed graph into output dir
python3 tools/preprocess.py kinships output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/kinships_gpu.ini info