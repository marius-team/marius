# Docker Installation

The following instructions install the necessary dependencies and build
the system using Docker. We describe the installation for GPU-based machines, 
although Marius and MariusGNN can run on CPU only machines as well.

### Build and Install Instructions ###
1. Check if docker is installed (`which docker`) and if not install it: https://docs.docker.com/engine/install/
2. Check if docker can access the GPUs by running `sudo docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`. If this doesn't print the output of `nvidia-smi`, docker cannot access the CUDA driver on the host machine and you need to install the NVIDIA drivers for GPU support.
3. Once the above succeeds, you should no longer need anything installed on the host machine.
4. Create a docker image using the provided Dockerfile: `docker build -t image_name:image_tag gpu_ubuntu/.`
5. Run the docker image: `docker run --gpus all -it image_name:image_tag bash`. It is often useful to link the current directory into the containers `/working_dir/` using the `-v` option (see below).
6. Once the container is running, install and build the system:
   ```
   cd marius
   pip3 install . --no-build-isolation
   ```

**Full List of Example Commands for GPU Installation**:

```
git clone https://github.com/marius-team/marius.git
cd marius
export CURRENT_DIR=`pwd`
cd examples/docker
docker build -t marius:latest gpu_ubuntu/.
docker run --gpus all -d -v $CURRENT_DIR:/working_dir/ --name=marius marius:latest sleep infinity
docker exec -it marius bash
pip3 install . --no-build-isolation
```

**CPU Only Installation**: If your machine does not have a GPU, remove the `--gpus all` from the docker run command in the GPU installation instructions. 
You can also optionally use the Dockerfile in `cpu_ubuntu/` rather than `gpu_ubuntu/`.

**Installation Notes**:
1. The installation requires Docker to have at least 8GB of memory to work with. This is generally satisfied by
   default, but if not (often on Mac), the `docker build` command may throw an error code 137. See
   [here](https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container/44533437#44533437),
   [here](https://stackoverflow.com/questions/34674325/error-build-process-returned-exit-code-137-during-docker-build-on-tutum), and
   [here](https://stackoverflow.com/questions/57291806/docker-build-failed-after-pip-installed-requirements-with-exit-code-137)
   for StackOverflow threads on how to increase Docker available memory or fix this issue. The `pip3 install .` command
   may also cause Docker memory issues. Increase the memory available to Docker or decrease the number of threads used for building
   MariusGNN (to decrease the number of threads change `-j{}` in line 45 of `setup.py` to `-j1` for example). One thread
   should build with 8GB of memory but may take some time (~30mins).