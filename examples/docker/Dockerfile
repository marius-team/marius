FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
RUN apt update

RUN apt install -y g++ \
         make \
         wget \
         unzip \
         vim \
         git \
         dstat \
         python3-pip

# install gcc-9
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9

# install cmake 3.20
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh cmake-3.20.0-linux-x86_64.sh --skip-license --prefix=/opt/cmake/
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# install pytorch
RUN python3 -m pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install DGL
RUN pip3 install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

# install PyG
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
RUN pip3 install pytorch-lightning

RUN mkdir /working_dir
WORKDIR /working_dir