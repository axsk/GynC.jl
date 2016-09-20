#__precompile__()

module GynC

const BATCHDIR = "/nfs/datanumerik/bzfsikor/batch"

import ForwardDiff # to compute derivative of objective


using Distributions
export WeightedChain
export emiteration!, euler_A!, euler_phih!

export Lausanne, Pfizer
export Config, Sampling
export sample, sample!, batch
export load, save
export samples


import DataFrames
include("data/lausanne.jl")
include("data/pfizer.jl")
include("gyncycle.jl")

import Sundials
include("model.jl")

import Mamba
include("sampling.jl")

export plotsolutions, plotdata



import JLD, HDF5
include("utils.jl")

include("batch.jl")



include("projectsimplex.jl")
include("weightedchain.jl")
include("priorestimation.jl")

#include("optim.jl")
include("optim2.jl")


include("plot.jl")

end
