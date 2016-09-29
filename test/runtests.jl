using GynC

seed = rand(1:1000)
info("seed: $seed")
srand(seed)

include("priorestimation.jl")

include("gync.jl")
