# Wikipedia Analysis

The following README contains the steps to perform the benchmarking on the wikipedia datasets. Before we run anything, run the commands:
```
$ sudo apt update -y && sudo apt upgrade -y
```

## Mounting the data directory

First run `lsblk` which produces an output like this:
```
NAME      MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
sda         8:0    0 447.1G  0 disk 
├─sda1      8:1    0   256M  0 part /boot/efi
├─sda2      8:2    0     1M  0 part 
├─sda3      8:3    0    64G  0 part /
└─sda99   259:2    0     8G  0 part [SWAP]
sdb         8:16   0 447.1G  0 disk 
sdc         8:32   0 745.2G  0 disk 
sdd         8:48   0 745.2G  0 disk 
sde         8:64   0 745.2G  0 disk 
sdf         8:80   0 745.2G  0 disk 
sdg         8:96   0 745.2G  0 disk 
sdh         8:112  0 745.2G  0 disk 
sdi         8:128  0 745.2G  0 disk 
sdj         8:144  0 745.2G  0 disk 
nvme0n1   259:1    0   1.5T  0 disk 
└─vg1-lv1 253:0    0   1.5T  0 lvm 
```

Then run the command:
```
$ mkdir -p all_data
```

The  update the `/etc/fstab` file to include the following line:
```
/dev/vg1/lv1  /users/sardev/all_data   xfs     defaults        0 0
```
but your path for the all_data directory might be different. 

Then mount the directory using the commands:
```
$ sudo mount -a
$ sudo chmod ugo+rw -R all_data
```

Verify by running `df -h` inside of `all_data` and ensure it produces this output:
```

```

## Setting up docker

First install the nvidia driver using the command:
```
$ sudo apt install -y nvidia-driver-550
```

Then install docker using the commands:
```
$ # Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
$ sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
$ sudo docker run hello-world
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker
```

Then install the GPU driver for containers using:
```
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
```

Finally run `sudo reboot`. Verify the install by running the command `nvidia-smi`. 

## Installing marius

Then install marius using the commands:
```
$ export CURRENT_DIR=`pwd`
$ git clone https://github.com/marius-team/marius.git
$ cd marius
$ git checkout -b dsarda/wikipedia
$ cd examples/docker/
$ docker build -t marius:latest gpu_ubuntu/.
$ docker kill marius
$ docker rm marius
$ docker run --gpus all -d -v $CURRENT_DIR:/root/ --name=marius marius:latest sleep infinity
$ docker exec -it marius bash
$ cd marius
$ python3 -m pip install "numpy<2" pybind11
$ pip3 install . --no-build-isolation
```

## Preprocessing graph snapshot

First setup aws using:
```
$ apt install -y python3-pip awscli
$ python3 -m pip install boto3
```

Then setup aws using `aws configure`. Then run the preprocessing using the commands:
```
$ cd wikipedia_analysis
$ python3 -u preprocess_runner.py &> preprocess.log
```

## Training the initial snapshot

Then train the initial snapshot using the file `initial_training.yaml`. Note that you might need to update the path of the datasets. Here is the command to run the training in (`marius`) by first running:
```
$ mkdir -p build && cd build
$ cmake ../ -DUSE_CUDA=TRUE -DUSE_OMP=TRUE
```

and then:
```
$ rm -rf /root/all_data/graph_snapshots/initial_snapshot/marius_formatted/model_* && make marius_train -j && ./marius_train ../wikipedia_analysis/initial_training.yaml
```