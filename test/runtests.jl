using GynC

sim = mcmc(GynC.GynCConfig(Lausanne(1)), 10)

include("priorestimation.jl")
include("io.jl")
