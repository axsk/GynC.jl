using GynC, Iterators

include("odes.jl")
include("utils.jl")

nsamples = 300; # number of samples in parameter space
ndata    = 2000; # number of simulated data points

rho_std = 5 # standarddeviation of the meas. error

xmin = 1; xmax = 110 
samples = linspace(xmin, xmax, nsamples) |> collect


zs = map(odeohnetreatment, samples)

function generatedata(n)
    k1 = 15
    k2 = 30
    s1 = 0.15*k1
    s2 = 0.15*k2
    n1 = floor(Int, n/2)
    x1 = k1 + s1*randn(n1)
    x2 = k2 + s2*randn(n-n1)

    xs = [x1; x2]
    zs = map(x->odeohnetreatment(x) + rho_std*randn(2), xs)
end

datas = generatedata(ndata)

g(x,mu,s)  = exp(-(x-mu).^2/(2*s^2))/sqrt(2*pi*s^2);
rho_error(x::Real) = g(x, 0, rho_std)
rho_error(x::Vector) = prod(map(rho_error, x))

zps = map(x->x + rho_std*randn(2), zs)

Lzd = [rho_error(z-d)  for d  in datas, z in zs]
Lzz = [rho_error(z-zp) for zp in zps,   z in zs]

marginallikelihood(w) = GynC.logLw(w, Lzd)
zentropy(w) = GynC.Hz(w, Lzz)

pml(w, s=1) = marginallikelihood(w) + s*zentropy(w)

## prior estimation

niter = 100
h  = 0.001
w0 = ones(nsamples) / nsamples

# weights after `niter` EM steps
function ws_em(w0, niter)
    take(iterate(w->GynC.emiteration(w, Lzd'), w0), niter) |> collect # todo unify L matrix orientation
end

# weights after `niter` pen.-max.-likelihood gradient-ascent steps
function ws_pml_ga(w0, niter, h=1)
    gradientascent(pml, w0, niter, h, GynC.projectsimplex)
end
