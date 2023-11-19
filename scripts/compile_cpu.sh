mkdir build_cpu
cd build_cpu

#cmake ../ -DUSE_CUDA=1
cmake ../ -DUSE_OMP=1
#cmake ../ -DUSE_CUDA=1 -DUSE_OMP=1 #-DCMAKE_C_COMPILER=clang
#cmake ../ -DUSE_CUDA=1 -DUSE_OMP=1 -DCMAKE_BUILD_TYPE=Debug -DMARIUS_USE_ASAN=1

make marius_train -j
#make marius_train marius_eval -j
#make _pymarius -j
#make unit -j

#make bindings -j
#mkdir marius
#cp *.so marius
#cp -r ../src/python/* marius/

#cp *.so /usr/local/lib/python3.6/dist-packages/marius/

cd ..