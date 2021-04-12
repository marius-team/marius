# preprocess the codex_m graph and put preprocessed graph into output dir
marius_preprocess codex_m output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/codex_m_cpu.ini info