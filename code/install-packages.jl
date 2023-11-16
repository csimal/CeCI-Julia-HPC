using Pkg

packages = [
    "IJulia",
    "PythonCall",
    "RCall",
    "BenchmarkTools",
    "ProfileView",
    "PProf",
    "LoopVectorization"
    "StatsPlots",
    "Dagger",
    "Folds",
    "Transducers",
]

for p in packages
    Pkg.add(p)
end
