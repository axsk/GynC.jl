using Distributions
import Distributions: length, insupport, logpdf, _logpdf


type UnivariateDensityDistribution <: ContinuousUnivariateDistribution
  pdf::Function
  insupport::Function
end

type MultivariateDensityDistribution <: ContinuousMultivariateDistribution
  pdf::Function
  dim::Integer
  insupport::Function
end

function DensityDistribution(n, f::Function, insupport::Function=((x)->true))
  # TODO: determine number of arguments
  n <= 1 ? 
    UnivariateDensityDistribution  (f, insupport) : 
    MultivariateDensityDistribution(f, n, insupport)
end

logpdf   (d::UnivariateDensityDistribution, x)   = log(d.pdf(x))
_logpdf  (d::MultivariateDensityDistribution, x) = log(d.pdf(x))
insupport(d::UnivariateDensityDistribution, x)   = d.insupport(x)
insupport(d::MultivariateDensityDistribution, x) = d.insupport(x)
length   (d::UnivariateDensityDistribution)      = 1
length   (d::MultivariateDensityDistribution)    = d.dim


""" Wrapper to the LIMEX solver for the GynC model 
solve the model for the times t given initial condition y0 and parameters parms, and store the result in y"""
function gync!(y::Array{Float64,2}, y0::Vector{Float64}, t::Vector{Float64}, parms::Vector{Float64})
  n = length(y0)
  m = length(t)
  ccall((:limstep_, "fortran/GynC.so"), Void,  
    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}),
    y, y0, t, &n, &m, parms)
end
