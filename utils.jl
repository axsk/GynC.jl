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
  true
end
