# preprocess the codex_l graph and put preprocessed graph into output dir
marius_preprocess codex_l output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/codex_l_cpu.ini info