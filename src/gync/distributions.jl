using Distributions
import Distributions: logpdf, minimum, maximum, length, insupport, _logpdf, size


type UnivariateDensityDistribution <: ContinuousUnivariateDistribution
  lpdf::Function
  min::Real
  max::Real
end

type MultivariateDensityDistribution <: ContinuousMultivariateDistribution
  lpdf::Function
  dim
  insupport::Function
end

function gaussianmixture(y::Matrix, stdfactor=1)
   stds = mapslices(std, y, 2) * stdfactor |> vec
   vars = abs2(stds)
   normals = mapslices(yt->MvNormal(yt, vars), y, 1) |> vec
   MixtureModel(normals)
end

# HOTFIX for missing insupport method for MixtureModel
insupport(d::Distributions.MultivariateMixture, x::AbstractVector) = true

""" Constructs a Distribution based on the given density function """
DensityDistribution(pdf::Function; kwargs...) = DensityDistribution(1, pdf; kwargs...)

function DensityDistribution(dim, pdf::Function; log=false, insupport::Function=((x)->true), intervall=[-Inf,Inf])
  lpdf = log ? pdf : x -> log(pdf(x)) 
  (isa(dim, Number) && dim <= 1) ? 
    UnivariateDensityDistribution(lpdf, intervall[1], intervall[2]) : 
    MultivariateDensityDistribution(lpdf, dim, insupport)
end
  
logpdf(d::UnivariateDensityDistribution, x::Real)        = d.lpdf(x)
minimum(d::UnivariateDensityDistribution)                = d.min
maximum(d::UnivariateDensityDistribution)                = d.max

logpdf(d::MultivariateDensityDistribution, x::DenseMatrix, transform::Bool=true) = d.lpdf(x)
#_logpdf(d::MultivariateDensityDistribution, x::DenseMatrix)   = d.lpdf(x)
insupport(d::MultivariateDensityDistribution, x::DenseMatrix) = d.insupport(x)
length(d::MultivariateDensityDistribution)               = d.dim
size(d::MultivariateDensityDistribution)                 = d.dim
