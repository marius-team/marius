# preprocess the codex_l graph and put preprocessed graph into output dir
python3 tools/preprocess.py codex_l output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/codex_l_multi_gpu.ini info