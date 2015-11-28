###  return truncated flat distributions on the cube product([0, max_i])

truncflatprior(max::Vector) = UnivariateDistribution[Truncated(Flat(),0,y) for y in max]


### Prior for y using the mixture of known values and their variances

import Distributions: minimum, maximum
minimum(d::Distributions.MixtureModel) = -Inf
maximum(d::Distributions.MixtureModel) =  Inf

independentmixtureprior(stdfactor::Real=1) = independentmixtureprior(mlegync(), stdfactor)

function independentmixtureprior(y::Matrix, stdfactor::Real)
  mms = mapslices(y, 2) do yi
    normals = [Normal(yit, std(yi) * stdfactor) for yit in yi]
    Truncated(MixtureModel(normals), 0, Inf)
  end
  UnivariateDistribution[mms...]
end

### Prior for y using full covariance of known values 

type GaussianMixtureDistr <: ContinuousMultivariateDistribution
  normals::Vector{MvNormal}
end

fullmixtureprior() = fullmixtureprior(mlegync())
function fullmixtureprior(y::Matrix)
  q = Base.cov(y, vardim=2)
  normals = mapslices(y_t->MvNormal(vec(y_t), q), y, 1) |> vec
  GaussianMixtureDistr(normals)
end

logpdf(d::GaussianMixtureDistr, x::Vector) = map(normal->pdf(normal, x), d.normals) |> mean |> log
insupport(d::GaussianMixtureDistr, x::Vector) = (r=all(x.>0); println(r); r)
length(d::GaussianMixtureDistr) = 33
