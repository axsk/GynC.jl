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
include("gync/sampling.jl")

import JLD, HDF5
include("gync/utils.jl")
include("gync/batch.jl")

include("gync/weightedchain.jl")
include("gync/plot.jl")


# Federn

export Federn
include("federn/federn.jl")


# EMPIRICIAL BAYES

export emiteration!, euler_A!, euler_phih!
export gradientascent
export Hz, logLw, HKL

include("eb/projectsimplex.jl")
include("eb/weightedsampling.jl")

include("eb/em.jl")

using Memoize
include("eb/regularizers.jl")
using Iterators
using ForwardDiff
include("eb/gradientascent.jl")

using Requires
@require NLopt include("eb/optim.jl")


end
