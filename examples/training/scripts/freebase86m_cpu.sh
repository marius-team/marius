# preprocess the freebase86m graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset freebase86m

# run marius on the preprocessed input
marius_train examples/training/configs/freebase86m_cpu.ini