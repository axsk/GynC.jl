module Federn

using Distributions
include("odes.jl")

phi = odeohnetreatment

const prior = MixtureModel([Normal(15, 15*.15), Normal(30, 30*.15)])

p_measerror(rho_std) = MvNormal(2, rho_std)
perturb(y, rho_std) = y + rand(p_measerror(rho_std))

#=function generatepriordata(n, rho_std)
    k1 = 15
    k2 = 30
    s1 = 0.15*k1
    s2 = 0.15*k2
    n1 = floor(Int, n/2)
    x1 = k1 + s1*randn(n1)
    x2 = k2 + s2*randn(n-n1)

    xs = [x1; x2]
    zs = perturb.(phi.(xs), rho_std)
end=#

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

function wbeta(xs,xmax) 
    beta(x) =  (x/xmax).*(1-x/xmax).^3.*(x/xmax.>0).*(x/xmax.<1)*20/xmax
    w = beta.(xs)
    w = w / sum(w)
end

using GynC: LikelihoodModel

function federmodel(nx, ndata, zmult, rho_std)
  xs, ys, datas, zs = federexperiment(nx=nx, ndata=ndata, zmult=zmult, rho_std=rho_std)
  LikelihoodModel(xs, ys, zs, datas, MvNormal(2, rho_std))
end

function betaprior(m::LikelihoodModel)
    wbeta(m.xs, maximum(m.xs)*1.000001) # to circumvent 0 weight destroying gradientascent boundary detection
end

end
