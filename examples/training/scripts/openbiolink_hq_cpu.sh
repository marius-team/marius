# preprocess the openbiolink_hq graph and put preprocessed graph into output dir
python3 tools/preprocess.py openbiolink_hq output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/openbiolink_hq_cpu.ini info