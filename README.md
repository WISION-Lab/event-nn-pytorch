## Conda Environment

To create the `event-nn` environment, run:
```
conda env create -f conda/environment.yml
```

To enable GPU support, instead run:
```
conda env create -f conda/environment_gpu.yml
```

## Code Style

Format all Python using [Black](https://black.readthedocs.io/en/stable/). Use a line limit of 88 characters (the default). To format a file, use the command:
```
black <FILE>
```

Format all C++ using [Clang-Format](https://clang.llvm.org/docs/ClangFormat.html). Use the default settings. To format a file, use the command:
```
clang-format -i <FILE>
```
