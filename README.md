#For CPU-only (Linux), download:

```link
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
```

#For GPU (Linux) with CUDA 12.1, use:
```link
https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
```

```bash
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip -d ~/libtorch
```
Set libtorch path
```bash
export LIBTORCH=/home/buddy/libtorch/libtorch
```
Set Library path
```bash
export LD_LIBRARY_PATH=/home/buddy/libtorch/libtorch/lib:$LD_LIBRARY_PATH
```# garbage_classification
