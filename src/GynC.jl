#__precompile__()

module GynC

const BATCHDIR = "/nfs/datanumerik/bzfsikor/batch"


# MCMC / GynC

export Lausanne, Pfizer
export Config, Sampling
export sample, sample!, batch
export load, save
export samples
export plotsolutions, plotdata
export WeightedChain

import DataFrames
include("gync/data/lausanne.jl")
include("gync/data/pfizer.jl")

using Distributions
import Sundials
include("gync/gyncycle.jl")
include("gync/model.jl")

import Mamba

import StatsBase: sample,sample! # strange fix, needed?

include("gync/sampling.jl")

import JLD, HDF5
include("gync/utils.jl")
include("gync/batch.jl")

include("gync/weightedchain.jl")
include("gync/plot.jl")

# empirical bayes

include("eb/eb.jl")

# Federn

export Federn
include("federn/federn.jl")

end
