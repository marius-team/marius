# preprocess the drkg graph and put preprocessed graph into output dir
python3 tools/preprocess.py drkg output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/drkg_cpu.ini info