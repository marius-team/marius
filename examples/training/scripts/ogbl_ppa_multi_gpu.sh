# preprocess the ogbl_ppa graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbl_ppa output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbl_ppa_multi_gpu.ini info