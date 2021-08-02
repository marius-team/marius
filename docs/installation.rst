.. _installation:

Dependencies
^^^^^^^^^^^^

(Other versions may work, but are currently untested)

* Ubuntu 18.04 or macOS 10.15
* CUDA 10.1 or 10.2 (If using GPU training)
* CuDNN 7 (If using GPU training)
* 1.7 >= pytorch
* python >=3.6
* pip >= 21
* GCC >= 9 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

Installing dependencies
"""""""""""""""""""""""
This can be skipped if the above are already installed.

PyTorch, CUDA & CuDNN:


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
