__precompile__()

module GynC

import ForwardDiff # to compute derivative of objective

include("projectsimplex.jl")
include("priorestimation.jl")
include("../examples/gc/gync.jl")

export WeightedChain
export emiteration!, euler_A!, euler_phih!

end
