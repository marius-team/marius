# preprocess the live_journal graph and put preprocessed graph into output dir
python3 tools/preprocess.py live_journal output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/live_journal_cpu.ini info