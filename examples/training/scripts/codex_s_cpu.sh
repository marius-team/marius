# preprocess the codex_s graph and put preprocessed graph into output dir
python3 tools/preprocess.py codex_s output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/codex_s_cpu.ini info