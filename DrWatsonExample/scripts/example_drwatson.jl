using DrWatson

@quickactivate "DrWatsonExample"

logistic_map(x, μ) = μ*x*(one(x)-x)

function logistic_sim(x₀::T, μ, niter) where {T<:Real}
    x = Vector{T}(undef, niter)
    x[1] = x₀
    for i in 2:niter
        x[i] = logistic_map(x[i-1], μ)
    end
    return x
end


allparams = Dict(
    "x₀" => collect(LinRange(0.0,1.0,10)),
    "μ" => collect(LinRange(0.0, 3.0, 100)),
    "niter" => 100,
    "name" => "logistic"
)

dicts = dict_list(allparams)

function makesim(d::Dict)
    @unpack x₀, μ, niter = d
    x = logistic_sim(x₀, μ, niter)
    fulld = copy(d)
    fulld["x"] = x
    return fulld
end

for (i,d) in enumerate(dicts)
    f = makesim(d)
    wsave(datadir("simulations", savename(d, "jld2")), f)
end
