# preprocess the fb15k_237 graph and put preprocessed graph into output dir
python3 tools/preprocess.py fb15k_237 output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/fb15k_237_gpu.ini info