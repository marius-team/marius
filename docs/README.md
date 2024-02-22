# Building the Docs #

1. Clone main repository: `git clone https://github.com/marius-team/marius.git`.

2. Clone `gh-pages` branch into seperate directory `html`: `git clone -b gh-pages https://github.com/marius-team/marius.git html`

3. Enter main repo: `cd marius`. Create build directory and run CMake with `BUILD_DOCS` enabled: `mkdir build; cd build; cmake ../ -DBUILD_DOCS=1`.

2. Build the documentation with Sphinx `make Sphinx -j`

3. Output html files will be generated in our `html` directory. Push changes to `gh-pages` for site to update at https://marius-project.org/marius/.