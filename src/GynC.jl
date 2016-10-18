#__precompile__()

module GynC

const BATCHDIR = "/nfs/datanumerik/bzfsikor/batch"


# MCMC

export Lausanne, Pfizer
export Config, Sampling
export sample, sample!, batch
export load, save
export samples
export plotsolutions, plotdata


import DataFrames
include("data/lausanne.jl")
include("data/pfizer.jl")

using Distributions
import Sundials
include("gyncycle.jl")
include("model.jl")

import Mamba
include("sampling.jl")

import JLD, HDF5
include("utils.jl")
include("batch.jl")


# EMPIRICIAL BAYES

export WeightedChain
export emiteration!, euler_A!, euler_phih!
export hzobj

include("projectsimplex.jl")
include("weightedchain.jl")
include("weightedsampling.jl")

include("em.jl")

import ForwardDiff # to compute derivative of objective
using Requires
@require NLopt include("optim.jl")
include("optim2.jl")
include("optim3.jl")


# PLOT

include("plot.jl")

end
