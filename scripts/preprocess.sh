export MARIUS_ONLY_PYTHON=1
export MARIUS_NO_BINDINGS=1

pip3 uninstall marius
pip3 install .

#mkdir ../datasets

#marius_preprocess --dataset fb15k_237 ../datasets/fb15k237/
#marius_preprocess --dataset fb15k_237 --num_partitions 16 ../datasets/fb15k237_partitioned/
#marius_preprocess --dataset fb15k_237 --num_partitions 128 ../datasets/fb15k237_partitioned_128/
#marius_preprocess --dataset fb15k_237 --num_partitions 256 ../datasets/fb15k237_partitioned_256/

#marius_preprocess --dataset live_journal ../datasets/livejournal/
#marius_preprocess --dataset live_journal --num_partitions 16 ../datasets/livejournal_16/

#marius_preprocess --dataset freebase86m ../datasets/freebase86m/
#marius_preprocess --dataset freebase86m --num_partitions 8 ../datasets/freebase86m_8/
#marius_preprocess --dataset freebase86m --num_partitions 16 ../datasets/freebase86m_16/
#marius_preprocess --dataset freebase86m --num_partitions 32 ../datasets/freebase86m_32/
#marius_preprocess --dataset freebase86m --num_partitions 64 ../datasets/freebase86m_64/
#marius_preprocess --dataset freebase86m --num_partitions 256 ../datasets/freebase86m_256/
#marius_preprocess --dataset freebase86m --num_partitions 1024 ../datasets/freebase86m_1024/
#marius_preprocess --dataset freebase86m --num_partitions 4096 ../datasets/freebase86m_4096/



#marius_preprocess --dataset cora ../datasets/cora/

#marius_preprocess --dataset OGBN_ARXIV ../datasets/arxiv/


#marius_preprocess --dataset fb15k_237 --output_dir datasets/fb15k_237_example/