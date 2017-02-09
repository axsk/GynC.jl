# EMPIRICIAL BAYES 
using Memoize
using Iterators
using ForwardDiff
using ReverseDiff
using NLopt

export emiteration!, euler_A!, euler_phih!
export gradientascent
export Hz, logLw, HKL

include("projectsimplex.jl")
include("weightedsampling.jl")

include("em.jl")

include("likelihoodmodel.jl")
include("likelihoodmat.jl")

include("regularizers.jl")

include("gradientascent.jl")
include("optim.jl")
include("linesearch.jl")
