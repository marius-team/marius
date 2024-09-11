ls
pwd
cd /root/
ls
numact --hardware
apt install -y numactl
numactl --hardware
apt-get install -y libnuma-dev
ls
cd numa_benchmarking/
g++ -o numa_test_runner numa_test.cpp -lnuma -std=c++9
g++ -o numa_test_runner numa_test.cpp -lnuma -std=c++11
ls
./numa_test_runner 
exit
exit
ls
clear
cd numa_benchmarking/
g++ -o numa_test_runner num_test.cpp -lnuma
ls
g++ -o numa_test_runner numa_test.cpp -lnuma
chmod ugo+x ./numa_test_runner 
./numa_test_runner 
exit
