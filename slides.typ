#import "@preview/polylux:0.3.1": *
#import themes.simple: *
//#import "theme-unamur.typ": *
#import "@preview/codetastic:0.2.2": qrcode
//#show: unamur-theme.with()
#show: simple-theme.with()

#set text(size: 20pt)
#show link: set text(blue)
#set text(font: "Comfortaa")
#show heading: set text(font: "Comfortaa")
#show raw: set text(font: "JuliaMono")
#show emph: set text(font: "Carlito")

#let note(dy: 1.5em, textsize: 0.65em, body) = {
    set text(size: textsize)
    place(bottom + left,
    dy: dy, body)
}

#title-slide[
    = Julia for HPC
    == Cédric Simal
    #link("cedric.simal@unamur.be")
    == CéCI Training Sessions
    20/11/2023
]

#centered-slide[
== Science Anniversaries of today
#side-by-side[
    #image("images/Zarya.jpeg", height: 70%)
    #text(size: 0.7em, [Zarya Launch (1998)])
][
    #image("images/Hubble.jpeg", height: 70%)
    #text(size: 0.7em, [Birth of Edwin Hubble (1889)])
    
][
    #image("images/Mandelbrot.jpeg", height: 70%)
    #text([Birth of Benoit Mandelbrot (1924)], size: 0.7em)
]

#place(bottom + left, 
dy: 1em,
text(size: 0.7em, [image source: Wikipedia]))
]

#slide[
== Outline
#set enum(start: 0)
#side-by-side[
+ Recap on Multiple Dispatch
+ Benchmarking and Profiling Julia code
+ Performance Tips
+ Interop with other languages
+ Parallelism in the Standard library
+ Packages for HPC
][
#set align(center)
Follow along!

#qrcode("https://github.com/csimal/CeCI-Julia-HPC")

#text(size:0.8em)[https://github.com/csimal/CeCI-Julia-HPC]
]
]


#focus-slide[
= Recap on Multiple Dispatch
]

#slide[
== Abstract and Concrete Types

#set align(center)
#image("images/type-tree-annotated.png", height: 80%)
]

#slide[
== Multiple Dispatch

Functions can be overloaded based on the types of all their arguments
#side-by-side[
```jl
foo(x) = "That doesn't look like a number"
foo(x::Real) = "Got a real number"
foo(x::Integer) = "Got an integer"
foo(x::Int64) = "Got a 64 bit integer"
foo(x::AbstractFloat) = "Got a floating point number"
```
][
```jl
bar(x,y) = ""
bar(x::Real, y::Real) = "Two real numbers"
bar(x::T, y::T) where {T<:Real} = "Two real numbers of the same type"
bar(x::Integer, x::Integer) = "Two integers"
bar(x::Integer, y::AbstractFloat) = "An integer and a float"
```
]
]

#slide[
== Some gotchas with parametric types
#grid(columns: (30%, 70%))[
```jl
struct Point2D{T}
    x::T
    y::T
end


```
][
```jl
# This will not work
function center(ps::Vector{Point2D{Real}})
    Point2D(
        mean(p.x for p in ps),
        mean(p.y for p in ps)
    )
end

# correct version
function center(ps::Vector{Point2D{<:Real}})
    Point2D(
        mean(p.x for p in ps),
        mean(p.y for p in ps)
    )
end
```
]
]

#slide[
== Some gotchas with parametric types
Explanation

```jl
Vector{Int} # a vector of Ints
Vector{T<:Real} where {T} # a vector of element type `T`, where `T` is a subtype of `Real`
Vector{T} where {T<:Real} # same as above
Vector{<:Real} # shorthand for above
Vector{Real} # a vector of mixed element types, all subtypes of `Real`

```
]

#focus-slide[
    = Benchmarking and Profiling
]

#slide[
== Benchmarking
#side-by-side[
Timing a single call
```jl
@time foo()
# See also
@timev foo() # verbose 
@elapsed foo() # just the time
@allocated foo() # just the memory used
```
][
For serious benchmarking
```jl
using BenchmarkTools

@benchmark foo()
# variants
@btime foo()
@belapsed foo()
@ballocated foo()
```
]
]

#slide[
== Profiling
#side-by-side[
Built-in Profiler
```jl
using Profile

@profile foo()



```
][
Visualizing profile data
```jl
using ProfileView

@profview foo()
```

```jl
using PProf

@pprof foo()
```
]

#note[https://docs.julialang.org/en/v1/stdlib/Profile/]
]


#focus-slide[
    = Performance Tips
    #note("https://docs.julialang.org/en/v1/manual/performance-tips/")
]

#slide[
== General Tips

- Put everything in functions
- Pre-allocate arrays, use in-place functions
- Global variables are evil. Use ```jl const```
- Use multiple dispatch to break up functions
- Access Arrays column by column
- Abstract types at runtime are the devil
]

#slide[
== Array views

#side-by-side[
```jl
function foo!(a)
    for i in eachindex(a)
        a[i] = sin(a[i])
    end
end

# this won't work
for i in axes(A,1)
    foo!(A[i,:])
    # NB. taking the slice `A[i,:]` creates a copy
end
```
][
Using views
```jl
for i in axes(A,1)
    foo!(view(A, i, :))
end

# convenient macro
@views for i in axes(A,1)
    foo!(A[i,:])
end
```
]
]

#slide[
== Type Stability
#side-by-side[
Don't do this...
```jl
relu(x) = x < 0 ? 0 : x

struct Foo
    x
    y::Real
    z::Vector{Integer}
end
```
][
but this
```jl
relu(x) = x < 0 ? zero(x) : x

struct Foo{T1,T2<:Real,T3<:Integer}
    x::T1
    y::T2
    z::Vector{T3}
end
```
]
Hint: Use the `@code_warntype` macro
]

#slide[
== Squeezing even more performance

- `StaticArray` (statically sized arrays, allocated on the stack)
- `@inline` hint that a function should be inlined
- `@inbounds` disable bounds checking on array indexing
- `@fastmath` (floating point optimizations)
- `@simd` use Single Instruction Multiple Data CPU ops within a loop
- `PackageCompiler.jl` Create pre-compiled artefacts

Note: Use with caution!

#note[
https://juliaarrays.github.io/StaticArrays.jl/stable/

https://julialang.github.io/PackageCompiler.jl/dev/index.html]
]

#focus-slide[
    = Interop with other languages
]

#slide[
== Calling Python/R
#side-by-side[
```jl
using PythonCall

re = pyimport("re")
words = re.findall("[a-zA-Z]+", "PythonCall.jl is very useful!")
sentence = Py(" ").join(words)
pyconvert(String, sentence)
```

See also `PyCall.jl`
][
```jl
using Rcall

x = randn(10)
R"t.test($x)"

jmtcars = reval("mtcars");
rcall(:dim, jmtcars)
```

]
#note[
https://juliapy.github.io/PythonCall.jl/stable/

https://juliainterop.github.io/RCall.jl/stable/gettingstarted/
]
]

#focus-slide[
    = Parallelism in the Standard Library
]

#slide[
== Multi-threading

#side-by-side[
Start Julia with multiple threads
```
julia --threads=4
```
Checking number of threads/thread id
```jl
Threads.nthreads()
Threads.threadid()
```
][
Running a loop over multiple threads
```jl
a = zeros(10)

Threads.@threads for i in eachindex(a)
    # your computation here
    a[i] = Threads.threadid()
end
```

]
#note[https://docs.julialang.org/en/v1/manual/multi-threading/]
]

#slide[
== Multithreading (2)
#set text(size: 0.8em)
Data races

#grid(columns: (30%,70%))[
```jl
function sum_single(a)
    s = 0
    for i in a
       s += i
    end
    s
end
```
][

```jl
function sum_multi_bad(a)
   s = 0
   Threads.@threads for i in a
       s += i
   end
   s
end

function sum_multi_good(a)
   chunks = Iterators.partition(a, length(a) ÷ Threads.nthreads())
   tasks = map(chunks) do chunk
       Threads.@spawn sum_single(chunk)
   end
   chunk_sums = fetch.(tasks)
   return sum_single(chunk_sums)
end

```
]

]

#slide[
== Distributed Computing
#side-by-side[
Start Julia with multiple processes
```
julia -p 4
```
or 
```jl
using Distributed
addprocs(4)
```
][
Low level interface
```jl
# call rand in processor 2 with argument (2,2)
r = remotecall(rand, 2, 2, 2)
# evaluate `1 .+ fetch(r)` in processor 2
s = @spawnat 2 1 .+ fetch(r)
# get the value of s
fetch(s)
# run expression on an arbitrary processor
t = @spawnat :any fetch(s)^2
```
]

#note[https://docs.julialang.org/en/v1/manual/distributed-computing/]
]

#slide[
== Loading dependencies accross processes
```jl
# define function on every process
@everywhere function foo()
    ...
end
# load package
@everywhere using LinearAlgebra
# load Julia file
@everywhere include("utils.jl")
```

Launch processes with dependencies
```
julia -p <n> -L file1.jl -L file2.jl main.jl
```
]

#slide[
== Distributed Loops
#side-by-side[
```jl
count = 0
for i in 1:n
    count += (norm(rand(2)) < 1.0)
end
```
Distributed (map)reduce
```jl
count = @distributed (+) for i in 1:n
    Int(norm(rand(2)) < 1.0)
end
```
][

Can be applied to outside variables
```jl
a = rand(1000000)
@distributed (+) for i in eachindex(a)
    a[i]
end
```
Parallel map
```jl
pmap(collatz, 1:10000)
```
]
]

#slide[
== Shared Arrays
#side-by-side[
This will not work
```jl
a = zeros(1000)
@distributed for i in eachindex(a)
    a[i] = i
end
```
][
Use a `SharedArray`
```jl
using SharedArrays

a = SharedArray{Int}(1000)
@distributed for i in eachindex(a)
    a[i] = i
end
```

]
#note[https://docs.julialang.org/en/v1/manual/distributed-computing/#man-shared-arrays]
]

#slide[
== Intermezzo: Switching algorithms using Multiple Dispatch
#set text(size: 0.85em)
#side-by-side[
```jl
function foo_single(n)
    for i in 1:n
        ...
    end
end

function foo_threaded(n)
    Threads.@threads for i in 1:n
        ...
    end
end

function foo_distributed(n)
    @distributed for i in 1:n
        ...
    end
end
```
][
Use empty types and multiple dispatch for user facing function
```jl
struct FooSingle end
struct FooThreaded end
struct FooDistributed end

foo(n) = foo(n, FooSingle())
foo(n, ::FooSingle) = foo_single(n)
foo(n, ::FooThreaded) = foo_threaded(n)
foo(n, ::FooDistributed) = foo_distributed(n)
```
]
]



#focus-slide[
    = Packages for HPC
]

#slide[
== Dagger.jl
#text(size: 0.8em)[
```jl
@everywhere begin
    using Random
    Random.seed!(0)

    # Some "expensive" functions that complete at different speeds
    const crn = abs.(randn(20, 7))
    f(i) = sleep(crn[i, 7])
    g(i, j, y) = sleep(crn[i, j])
end
function nested_dagger()
    @sync for i in 1:20
        y = Dagger.@spawn f(i)
        for j in 1:6
            z = Dagger.@spawn g(i, j, y)
        end
    end
end
```]
#note[https://juliaparallel.org/Dagger.jl/dev/]
]

#slide[
== Transducers.jl
Functional Style chains of operations
```jl
using Folds, Transducers

# runs in threaded mode by default
# NB. '|>' is the pipe operator
1:N |> Filter(isprime) |> Filter(ispalindromic) |> Map(n -> 1) |> Folds.sum

# distributed fold
foldxd(+, (1 for n in 1:N if isprime(n) && ispalindromic(n)))
```

#note[https://juliafolds.github.io/Transducers.jl/dev/]
]

#slide[
== GPUs
#side-by-side[
Dedicated packages
- `CUDA.jl` (run Julia natively on Nvidia GPUs)
- `oneAPI.jl`
- `AMDGPU.jl` (ROCm)
- `Metal.jl` (MacOS - WIP)

High level libraries
- `Tullio.jl` (einsum operations)
][
CUDA kernel
#set text(size: 0.75em)
```jl
function gpu_add!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)
fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
```
]
#note[https://cuda.juliagpu.org/stable/tutorials/introduction/]
]

#slide[
== Backend agnostic GPU kernels
#set text(size: 0.9em)
#side-by-side[
    ```jl
using KernelAbstractions

@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end
```
][
```jl
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(a, b, c, ndrange=size(c)) 
end
```
]
#note[
    https://juliagpu.github.io/KernelAbstractions.jl/stable/examples/matmul/

    https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html
]
]

#slide[
== Honorable mentions

- `MPI.jl`
- `ClusterManagers.jl` Running slurm jobs directly from Julia
- `DrWatson.jl` Setting up scientific projects
- `libblastrampoline` Switch the BLAS/LAPACK backend dynamically at runtime

]