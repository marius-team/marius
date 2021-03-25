# preprocess the ogbl_biokg graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbl_biokg output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbl_biokg_gpu.ini info