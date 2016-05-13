using GynC

sim = mcmc(ModelConfig(Lausanne(1)), 10)

include("reweight.jl")
include("io.jl")
