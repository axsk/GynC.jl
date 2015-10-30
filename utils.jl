using Distributions
import Distributions: logpdf, minimum, maximum, length, insupport, _logpdf


type UnivariateDensityDistribution <: ContinuousUnivariateDistribution
  pdf::Function
end

type MultivariateDensityDistribution <: ContinuousMultivariateDistribution
  pdf::Function
  dim::Integer
  insupport::Function
end

""" Constructs a Distribution based on the given density function """
function DensityDistribution(f::Function, insupport::Function=((x)->true))
  dim = length(Base.uncompressed_ast(f.code).args[1])
  dim <= 1 ? 
    UnivariateDensityDistribution(f) : 
    MultivariateDensityDistribution(f, dim, insupport)
end

logpdf(d::UnivariateDensityDistribution, x::Real)        = log(d.pdf(x))
minimum(d::UnivariateDensityDistribution)                = 0
maximum(d::UnivariateDensityDistribution)                = 200

_logpdf(d::MultivariateDensityDistribution, x::Vector)   = log(d.pdf(x))
insupport(d::MultivariateDensityDistribution, x::Vector) = d.insupport(x)
length(d::MultivariateDensityDistribution)               = d.dim

""" Given samplings (of the same size), concatenate them to form their mean sampling """ 
function mean(chains::Vector{Mamba.ModelChains}) 
  for i = 1:length(chains)-1
    size(chains[i], 1) == size(chains[i+1], 1) ||
      warn("concatenated chains have not same length")
  end
  Chains(cat(1, [c.value for c in chains]...))
end


""" Wrapper to the LIMEX solver for the GynC model 
solve the model for the times t given initial condition y0 and parameters parms, and store the result in y"""
function gync!(y::Array{Float64,2}, y0::Vector{Float64}, t::Vector{Float64}, parms::Vector{Float64})
  n = length(y0)
  m = length(t)
  ccall((:limstep_, "fortran/GynC.so"), Void,  
    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}),
    y, y0, t, &n, &m, parms)
end
