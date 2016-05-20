module GynC

using Mamba, Distributions
using JLD, HDF5
using DataFrames
using Requires

import Sundials
import ForwardDiff # to compute derivative of objective



include("projectsimplex.jl")
include("priorestimation.jl")
include("gync/gync.jl")

export WeightedChain
export emiteration!, euler_A!, euler_phih!

end
