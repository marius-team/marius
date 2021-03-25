# preprocess the wn18rr graph and put preprocessed graph into output dir
python3 tools/preprocess.py wn18rr output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/wn18rr_cpu.ini info