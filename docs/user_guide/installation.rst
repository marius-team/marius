.. _installation:

***************
Installation
***************

Dependencies
^^^^^^^^^^^^

(Other versions may work, but are currently untested)

* Ubuntu 18.04 or macOS 10.15
* CUDA 10.1 or 10.2 (If using GPU training)
* CuDNN 7 (If using GPU training)
* pytorch >= 1.7
* python >=3.6
* pip >= 21
* GCC >= 9 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

Installing torch according to their documentation should cover the above dependencies: https://pytorch.org/get-started/locally/


Installation from source with Pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Install latest version of PyTorch for your CUDA version:

    Linux:

    * CUDA 10.1: ``python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html``
    * CUDA 10.2: ``python3 -m pip install torch==1.7.1``
    * CPU Only: ``python3 -m pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html``

    MacOS:

    * CPU Only: ``python3 -m pip install torch==1.7.1``

#. Clone the repository ``git clone https://github.com/marius-team/marius.git``
#. Build and install Marius ``cd marius; python3 -m pip install .``

Script
"""""""""""""""""""""""""""""""""""

::

    git clone https://github.com/marius-team/marius.git
    cd marius
    python3 -m pip install .


Installation from source with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Clone the repository ``git clone https://github.com/marius-team/marius.git``
#. Install dependencies ``cd marius; python3 -m pip install -r requirements.txt``
#. Create build directory ``mkdir build; cd build``
#. Run cmake in the build directory ``cmake ../`` (CPU-only build) or ``cmake ../ -DUSE_CUDA=1`` (GPU build)
#. Make the Marius executable ``make marius_train -j``

Script
"""""""""""""""""""""""""""""""""""

::

    git clone https://github.com/marius-team/marius.git
    cd marius
    python3 -m pip install -r requirements.txt
    mkdir build
    cd build
    cmake ../ -DUSE_CUDA=1
    make -j

Marius in Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marius can be deployed within a docker container for convenient dependency management. Here is a sample ubuntu dockerfile which contains the necessary dependencies preinstalled for GPU training.

::

    FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
    RUN apt update

    RUN apt install -y g++ \
             make \
             wget \
             unzip \
             vim \
             git \
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
    RUN python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
