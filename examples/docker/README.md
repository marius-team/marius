# Sample dockerfile

Build an image with the name `marius` and the tag `example`:  
`docker build -t marius:gpu -f examples/docker/gpu_ubuntu/dockerfile examples/docker/gpu_ubuntu/`

Create and start a new container instance named `gaius` with:  
`docker run --name marius_gpu -itd marius:gpu`

Run `docker ps` to verify the container is running

Start a bash session inside the container:  
`docker exec -it marius_gpu bash`