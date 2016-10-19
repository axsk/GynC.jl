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
include("GynC/data/lausanne.jl")
include("GynC/data/pfizer.jl")

using Distributions
import Sundials
include("GynC/gyncycle.jl")
include("GynC/model.jl")

import Mamba
include("GynC/sampling.jl")

import JLD, HDF5
include("GynC/utils.jl")
include("GynC/batch.jl")

include("GynC/weightedchain.jl")
include("GynC/plot.jl")


# Federn

include("Federn/federn.jl")


# EMPIRICIAL BAYES

export emiteration!, euler_A!, euler_phih!
export hzobj

include("EB/projectsimplex.jl")
include("EB/weightedsampling.jl")

include("EB/em.jl")

using Memoize
include("EB/regularizers.jl")

import ForwardDiff # to compute derivative of objective
using Requires
@require NLopt include("EB/optim.jl")


end
