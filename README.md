# CéCI Training Days: Introduction to Julia

Julia is a recent (2012) programming language created at the MIT with the aim to provide performance on par with the likes of C and Fortran, with the ease of use of languages like Python and Matlab. The goal of this advanced session is to go deeper into the various tools for High Performance Computing in Julia, from multi-threaded to distributed computing, as well as the necessary tools to work with them, such as benchmarking and profiling code.

Contents:

- Benchmarking and Profiling Julia code
- Performance Tips
- Interop with other languages
- Parallelism in the Standard Library
- Packages for HPC


## Running the examples in this repo

You'll first need to install Julia on your computer. To do that, the easiest way is to use [juliaup](https://github.com/JuliaLang/juliaup) to install the latest version (v1.9.4).

To run the notebooks, you'll at least need a Jupyter installation. The easiest way for that would be to install [VSCode](https://code.visualstudio.com/) (or it's fully open source fork [VSCodium](https://vscodium.com/)), and then install the [Julia Extension](https://www.julia-vscode.org/docs/dev/gettingstarted/) along with the [Jupyter extension](https://github.com/microsoft/vscode-jupyter) (Both of these are available from the Plugin tab in VSCode)

Alternatively, you can just install Jupyter (via `pip`) and install the Julia Jupyter kernel. To do that, launch Julia in a terminal and enter the following line
```
] add IJulia
```
Once the package is installed, you'll be able to select Julia as a kernel in Jupyter notebooks.

## Learning Julia

### Webpages

- Official Manual https://docs.julialang.org/en/v1/
- https://julialang.org/learning/
- Noteworthy Differences from other languages https://docs.julialang.org/en/v1/manual/noteworthy-differences/

## Links included in the slides

- Profiling https://docs.julialang.org/en/v1.9/manual/profile/
- Performance Tips https://docs.julialang.org/en/v1.9/manual/performance-tips/
- PackageCompiler.jl https://julialang.github.io/PackageCompiler.jl/dev/index.html
- PythonCall.jl https://juliapy.github.io/PythonCall.jl/stable/
- PyCall.jl https://github.com/JuliaPy/PyCall.jl
- RCall.jl https://juliainterop.github.io/RCall.jl/stable/gettingstarted/
- Multi-threading https://docs.julialang.org/en/v1.9/manual/multi-threading/
- Distributed Computing https://docs.julialang.org/en/v1.9/manual/distributed-computing/
- Dagger.jl https://juliaparallel.org/Dagger.jl/dev/
- Transducers.jl https://juliafolds.github.io/Transducers.jl/dev/
- CUDA.jl https://cuda.juliagpu.org/stable/
- KernelAbstractions.jl https://juliagpu.github.io/KernelAbstractions.jl/stable/
- https://juliagpu.github.io/KernelAbstractions.jl/stable/examples/matmul/
- Porting a Fluid simulation solver to GPU https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html
- Tullio.jl https://github.com/mcabbott/Tullio.jl
- MPI.jl https://juliaparallel.org/MPI.jl/stable/
- ClusterManagers.jl https://github.com/JuliaParallel/ClusterManagers.jl
- DrWatson.jl https://juliadynamics.github.io/DrWatson.jl/dev/
- libblastrampoline https://github.com/JuliaLinearAlgebra/libblastrampoline



## Details about this repository

This repository is licensed under the Creative Commons CC0 Universal License.

The slides were made with [Typst](https://typst.app/). Demos were made using Jupyter notebooks.

All demos use Julia 1.9.4.