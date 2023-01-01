# Tensor Core Project

## File Structure

The main part of code is placed in `resnet` folder.

```
resnet
|- include: All header files
	|- functional: Functional primitives to perform Resnet18 inference
	|- modules: Resnet18 module components, wrapper of implemented Tensor class
	|- simulator: Simulator files
	other files: dataset loading, memory pool, tensor operations, etc.
|- src: Source files
	|- functional, modules, simulator: Definitions corresponding to the above headers
	|- cmake: Used for testing in CMake configuration
	|- python_helpers: Python script for weight and dataset preprocessing
	resnet_inference.cpp: Entrypoint for simulator and main inference
	other files: Definitions corresponding to the above headers
|- tests: unit test files
CMakeLists.txt: CMake script for configuration and build
*.cmake: Id.
doxygen.conf.in: Documentation generation file
```

## Compilation Guide

### Prepare Dataset and Model Weights

The main functionality only requires CUDA library to be installed, and will not depend on the following dependencies. You may check the compilation result with `ldd`:

```bash
$ ldd main
        linux-vdso.so.1 (0x00007ffcdb2f4000)
        libcudart.so.11.0 => /opt/cuda/lib64/libcudart.so.11.0 (0x00007f25a0200000)
        libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f259fe00000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f25a0118000)
        libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007f25a00f8000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f259fc19000)
        /lib64/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2 (0x00007f25a0523000)
        libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f25a04be000)
        libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f25a04b9000)
        librt.so.1 => /usr/lib/librt.so.1 (0x00007f25a04b4000)
```

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

### Available CMake Variables

Here are a list of CMake variables which can be used to toggle the compilation. 

For example a `EXAMPLE` variable, you can use `cmake -DEXAMPLE:BOOL=<ON/OFF>` to modify them.

```
DEBUG_ONE_ROUND Only run one round of inference (a batch of 16). Default: OFF
GENERATE_DATASET Generate dataset for testing and inference. Default: ON
GENERATE_DOCS Generate documentation. Default: OFF
ENABLE_SOMA Use CUDA Stream Ordered Memory Allocator. Default: ON
```

### Compile

Enter the `resnet/build` directory, run `cmake ..` to configure the project. Then run `make` to compile the project.

Check the output of CMake thoroughly. If there are dependencies missing, not all functionalities will be compiled.


## Run Guide

After compilation, the executable `main` and `main_simulator` will be generated in the `resnet/build` directory. 

The `main` executable is the main functionality of the project. It will run the ResNet-18 model on the ImageNet dataset.

The `main_simulator` executable performs the same function, but changed the GEMM to the simulated. It will simulate the performance of the Tensor Core based GEMM.

Both programs require no arguments. The dataset and model path, if correctly configured, will be automatically filled by C macros. This is intended as all data is preprocessed in a project-specific way.

To run unit tests, execute `gtest_main` in the `resnet/build` directory. 
