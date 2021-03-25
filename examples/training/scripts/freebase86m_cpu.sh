# preprocess the freebase86m graph and put preprocessed graph into output dir
python3 tools/preprocess.py freebase86m output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/freebase86m_cpu.ini info