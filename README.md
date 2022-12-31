# Tensor Core Project

## Compilation Guide

### Prepare Dataset and Model Weights

The main functionality only requires CUDA library to be installed.

 The ImageNet dataset should be put in `dataset/imagenet.tar.gz`, which needs to be grouped as:
 
 ```
 [imagenet.tar.gz]
 |- <A folder containing all images>
    |- <some>.JPEG
    |- <some_other>.JPEG
    |- ...
 ```
 
 CMake will prepare the dataset and weight automatically. A Python installation with following packages is required:
 
 - Python 3
 - PyTorch
 - torchvision
 - numpy
 - tqdm
 - PIL
 
 The environment can be configured with Conda. A conda environment package list is also provided in `conda.requirements.txt`.

 The module weight will be downloaded automatically by torchvision. 

 By entering the `resnet/build` directory, run `cmake ..`, it will automatically prepare the dataset and weights.

### Prepare test dependency

You need to install Google Test to run the unit tests. If you prepared so, after CMake configured, you can run `ctest` or `make test` to run all the tests. 

The tests invoked will be dependent on what packages you have installed. To run full tests, you need to install:

- CUDA Toolkit (for cuBLAS based GEMM correctness unit tests)
- libTorch (for libTorch based common layer correctness unit tests)

### Prepare reference documentation dependency

To generate the reference documentation, you need to install Doxygen. If you prepared so, set CMake option `GENERATE_DOCS` to ON, then you can run `make doc` to generate the documentation.

### Preconfigured Environment

For TA's convenience, the environment is preconfigured if you log into the provided server with user `group2`, and run the following commands:

```bash
conda activate with_cudatoolkit
```

Compatible versions of `cmake`, `gtest` and `libtorch` is also installed in `~/.local`.

The environment will be prepared to use. If you extracted a submitted version to `/path/to/submitted/root`, link the dataset directory to the root directory:

```bash
ln -s ~/dataset /path/to/submitted/root/dataset
```

The CMake configuration and compilation will be ready to go.

### Compile

Enter the `resnet/build` directory, run `cmake ..` to configure the project. Then run `make` to compile the project.

Check the output of CMake thoroughly. If there are dependencies missing, not all functionalities will be compiled.


### Run

After compilation, the executable `main` and `main_simulator` will be generated in the `resnet/build` directory. 

The `main` executable is the main functionality of the project. It will run the ResNet-18 model on the ImageNet dataset.

The `main_simulator` executable performs the same function, but changed the GEMM to the simulated. It will simulate the performance of the Tensor Core based GEMM.

Both programs require no arguments. The dataset and model path, if correctly configured, will be automatically filled by C macros. This is intended as all data is preprocessed in a project-specific way.
