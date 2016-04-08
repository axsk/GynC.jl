using GynC

sim = mcmc(ModelConfig(Lausanne(1)), 10)

include("io.jl")
include("reweight.jl")
