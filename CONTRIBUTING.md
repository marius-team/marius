# Contributing to Marius

Any contributions users wish to make to Marius are welcome. To name a few, here are some ways to contribute:


- Adding new models
- Adding new datasets and converters
- Downstream inference examples
- Documentation improvements
- Bug Fixes

## Contributing Code

1. Fork the Marius repository
2. Clone the forked repo and create a new branch for your change  
- `git clone https://github.com/<YourUsername>/marius`  
- `git checkout -b <feature_branch>`
   
3. Add your changes to the feature branch 

4. Write tests for your changes  
- C++ Tests are located in a gtest under `test/cpp`
- Python tests are located in `test/python`

5. Run tests and verify nothing is broken.
See the testing README for how to build and run the tests `test/README.md`

## Submitting a Pull Request

Once your changes have been completed, or if you want to submit an in-progress pull request to get eyes on it. Please follow the following steps:

1. Sync your feature branch with the main branch

- `git remote add upstream https://github.com/marius-team/marius.git`

- `git fetch upstream main`

- `git merge upstream/main`

2. Create and submit a pull request that follows the provided template. The pull request will be reviewed by the maintainers of Marius.

3. Address the comments from the reviewer(s) and update your pull request accordingly. 

4. Once the review process is complete your changes will be merged in!
