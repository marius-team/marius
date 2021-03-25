# preprocess the fb15k graph and put preprocessed graph into output dir
python3 tools/preprocess.py fb15k output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/fb15k_gpu.ini info