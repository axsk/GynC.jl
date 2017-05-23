module Federn

using Distributions
include("odes.jl")

phi(x) = odesol(x, [1, 1.7] * sqrt(6/.7), m=6)

const prior = MixtureModel([Normal(15, 15*.15), Normal(30, 30*.15)])

function perturb(y::Vector, rho) 
  err = MvNormal(length(y), rho)
  y + rand(err)
end


function generatepriordata(n, rho_std)
    xs = rand(prior, n)
    zs = map(x->perturb(phi(x), rho_std), xs)
end

function federexperiment(;
                         nx = 300, # number of samples in parameter space
                         ndata    = 2000, # number of simulated data points
                         rho_std = 5, # standarddeviation of the meas. error
                         xmin = 1, xmax = 110,
                         zmult = 1) # multiplier for z measurements in hz computation)

    xs = linspace(xmin, xmax, nx) |> collect

    ys = phi.(xs)

    datas = generatepriordata(ndata, rho_std)

    zs = perturb.(repmat(ys, zmult), rho_std)

    xs, ys, datas, zs
end

using GynC: LikelihoodModel

function wbeta(xs,xmax=maximum(xs))
    beta(x) =  (x/xmax).*(1-x/xmax).^3.*(x/xmax.>0).*(x/xmax.<1)*20/xmax
    w = beta.(xs)
    w = w / sum(w)
end

function federmodel(nx, ndata, zmult, rho_std; kwargs...)
  xs, ys, datas, zs = federexperiment(nx=nx, ndata=ndata, zmult=zmult, rho_std=rho_std; kwargs...)
  warn("check if MvNormal(2,...) is what you want")
  LikelihoodModel(xs, ys, zs, datas, MvNormal(2, rho_std))
end

function federmodel(;nx=100, ts=[1, 1.7] * sqrt(6/.7), ndata=0, zmult=20, rho=1, xmin=1, xmax=110, m=6)

  err = MvNormal(length(ts), rho)
  perturb(x) = x+rand(err)
  phi(x) = odesol(x, ts, m=m)

  xs = linspace(xmin, xmax, nx) |> collect
  ys = phi.(xs)
  zs = perturb.(repmat(ys, zmult))
  datas = perturb.(phi.(rand(prior, ndata)))

  LikelihoodModel(xs, ys, zs, datas, err)
end

betaprior(m::LikelihoodModel) = wbeta(m.xs, maximum(m.xs)*1.000001) # to circumvent 0 weight destroying gradientascent boundary detection

end
