using GynC

sim = mcmc(ModelConfig(Lausanne(1)), 10)

include("priorestimation.jl")
include("io.jl")
