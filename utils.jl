using Distributions
import Distributions: length, insupport, logpdf
export DensityDistribution

type DensityDistribution <: ContinuousUnivariateDistribution
  pdff::Function

  function DensityDistribution(f::Any)
    new(convert(Function, f))
  end
end

function logpdf(d::DensityDistribution, x::Real)
  log(d.pdff(x))
end

length(d::DensityDistribution) = 1

function insupport(d::DensityDistribution, x::Real)
  #TODO: implement bound support
  true
end

function gync(yall::Array{Float64,2}, y0::Vector{Float64}, tspan::Vector{Float64}, Parms::Vector{Float64})
  n = length(y0)
  m = length(tspan)
  ccall((:limstep_, "fortran/GynC.so"), Ptr{Array{Float64,2}}, 
    (Ptr{Float64}, 
     Ptr{Float64}, 
     Ptr{Float64}, 
     Ptr{Int64}, 
     Ptr{Int64}, 
     Ptr{Float64}),
    yall, y0, tspan, &n, &m, Parms)
end


  