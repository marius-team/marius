# preprocess the yago_310 graph and put preprocessed graph into output dir
python3 tools/preprocess.py yago_310 output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/yago_310_multi_gpu.ini info