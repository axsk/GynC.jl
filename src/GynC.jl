#__precompile__()

module GynC

using Requires

const BATCHDIR = "/nfs/datanumerik/bzfsikor/batch"

export WeightedChain
export emiteration!, euler_A!, euler_phih!

export Lausanne, Pfizer
export Config, Sampling
export sample, sample!, batch
export load, save
export samples
export plotsolutions, plotdata

import DataFrames
include("data/lausanne.jl")
include("data/pfizer.jl")
include("gyncycle.jl")

using Distributions
import Sundials
include("model.jl")

import Mamba
include("sampling.jl")



import JLD, HDF5
include("utils.jl")

include("batch.jl")


# EMPIRICIAL BAYES

import ForwardDiff # to compute derivative of objective

include("projectsimplex.jl")
include("weightedchain.jl")
include("priorestimation.jl")

@require NLopt include("optim.jl")
export hzobj
include("optim2.jl")

# PLOT

include("plot.jl")

end
