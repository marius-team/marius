# preprocess the cskg graph and put preprocessed graph into output dir
python3 tools/preprocess.py cskg output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/cskg_cpu.ini info