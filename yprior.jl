include("gyncycle.jl")

YPrior() = YPrior(mlegync())

type YPrior <: ContinuousMultivariateDistribution
  normals::Vector{MvNormal}
  scales::Vector  # needed for stability transformation
end

function YPrior(y::Matrix) 
  scales = mean(y, 2) |> vec
  scales *= 1/2
  invscalemat = scales.^-1 |> diagm
  q = Base.cov(invscalemat * y, vardim=2)
  normals = mapslices(y_t->MvNormal(vec(y_t./scales), q), y, 1) |> vec
  YPrior(normals, scales)
end


logpdf(d::YPrior, x::Vector) = map(normal->pdf(normal, x./d.scales), d.normals) |> sum |> log
insupport(d::YPrior, x::Vector) = all(x>.0)
length(d::YPrior) = length(d.normals)
