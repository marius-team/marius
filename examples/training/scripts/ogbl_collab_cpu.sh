# preprocess the ogbl_collab graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbl_collab output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbl_collab_cpu.ini info