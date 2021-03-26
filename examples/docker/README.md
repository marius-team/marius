# Sample dockerfile

Build an image with the name `marius` and the tag `example`:  
`docker build -t marius:example -f examples/docker/dockerfile examples/docker`

Create and start a new container instance named `gaius` with:  
`docker run --name gaius -itd marius:example`

Run `docker ps` to verify the container is running

Start a bash session inside the container:  
`docker exec -it gaius bash`