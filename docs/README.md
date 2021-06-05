# Building the Docs #

1. Run CMake with `BUILD_DOCS` enabled. `cd build; cmake ../ -DBUILD_DOCS=1`

2. Build the documentation with Sphinx `make Sphinx -j`

3. Open `build/docs/sphinx/index.html` in a web browser to view the docs.