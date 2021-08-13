# preprocess the codex_s graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset codex_s

# run marius on the preprocessed input
marius_train examples/training/configs/codex_s_multi_gpu.ini