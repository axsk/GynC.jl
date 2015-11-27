type GaussianMixtureDistr <: ContinuousMultivariateDistribution
  normals::Vector{MvNormal}
end

independentmixtureprior() = independentmixtureprior(mlegync())

""" given y[i,t], return the mixture of gaussians around each y[:,t] with diagonal variance of y[i,:] """
function independentmixtureprior(y::Matrix)
  stds    = std(y, 2) |> vec
  normals = mapslices(y, 1) do yt
    MvNormal(vec(yt), stds)
  end |> vec
  GaussianMixtureDistr(normals)
end


YPrior() = YPrior(mlegync())

function YPrior(y::Matrix) 
  scales = mean(y, 2) |> vec
  scales *= 1/2
  invscalemat = scales.^-1 |> diagm
  q = Base.cov(invscalemat * y, vardim=2)
  normals = mapslices(y_t->MvNormal(vec(y_t./scales), q), y, 1) |> vec
  YPrior(normals, scales)
end


logpdf(d::GaussianMixtureDistr, x::Vector) = map(normal->pdf(normal, x), d.normals) |> sum |> log
insupport(d::GaussianMixtureDistr, x::Vector) = all(x>.0)
length(d::GaussianMixtureDistr) = length(d.normals)
