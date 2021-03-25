# preprocess the wn18 graph and put preprocessed graph into output dir
python3 tools/preprocess.py wn18 output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/wn18_gpu.ini info