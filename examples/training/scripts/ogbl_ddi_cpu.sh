# preprocess the ogbl_ddi graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbl_ddi output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbl_ddi_cpu.ini info