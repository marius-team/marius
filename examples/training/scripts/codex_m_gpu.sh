# preprocess the codex_m graph and put preprocessed graph into output dir
python3 tools/preprocess.py codex_m output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/codex_m_gpu.ini info