# preprocess the codex_s graph and put preprocessed graph into output dir
marius_preprocess codex_s output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/codex_s_gpu.ini info