# Testing Marius #

Marius uses GTest for testing C++ code, and uses tox and pytest for testing python code.

Currently only a simple set of end to end tests are written for C++ and Python. 

### C++ Tests ###

Tests must be built before they can be run.

Building the tests (working from `<INSTALL_DIR>/marius`):  
```
cd build
make end_to_end -j
```

Running the tests: 
```
cd build/test/cpp/end_to_end
./end_to_end
```

### Python Tests ###

Running the tests (working from `<INSTALL_DIR>/marius`):
```
tox
```
