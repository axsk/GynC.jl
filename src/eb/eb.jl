# EMPIRICIAL BAYES 
using Memoize
using Iterators
using ForwardDiff
using ReverseDiff
using NLopt

export emiteration!, euler_A!, euler_phih!
export gradientascent
export Hz, logLw, HKL

include("eb/projectsimplex.jl")
include("eb/weightedsampling.jl")

include("eb/em.jl")

include("eb/likelihoodmodel.jl")
include("eb/likelihoodmat.jl")

include("eb/regularizers.jl")

include("eb/gradientascent.jl")
include("eb/optim.jl")
include("eb/linesearch.jl")
