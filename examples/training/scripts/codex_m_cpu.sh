# preprocess the codex_m graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset codex_m

# run marius on the preprocessed input
marius_train examples/training/configs/codex_m_cpu.ini