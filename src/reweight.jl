abstract WeightedChain

type SimpleWeightedChain <: WeightedChain
  weights::AbstractVector
  likelihoods::AbstractMatrix # row = param, col = subject
end


### GynCChain constructors for the GynC model ###
# TODO: move this section to model.jl / create a WeightedMambaChain

type GynCChain <: WeightedChain
  chain::AbstractMatrix
  weights::AbstractVector
  likelihoods::AbstractMatrix
end

" construct the GynCChain computing the likelihoods for the given `samples` (row = sample, col = sampledparam) given `datas` with error `sigma` "
function GynCChain(chain::Matrix, datas::Vector{Matrix}, sigma::Real)
  GynCChain(chain, ones(size(chain, 1)), likelihoods(chain, datas, sigma))
end
GynCChain(c::Vector, w, l) = GynCChain(reshape(c,length(c),1), w, l)

" compute the likelihood matrix for given chains, data, sigma) "
function likelihoods(chain::AbstractMatrix, data::Vector{Matrix}, sigma::Real)
  K = size(chain, 1)
  M = length(data)
  likelihoods = SharedArray(Float64,K,M)
  @sync @parallel for k = 1:K
    for m = 1:M
      likelihoods[k,m] = likelihood(data[m], chain[k,:]|>vec, sigma)
    end
  end
  likelihoods
end

# TODO: base computation on model
" compute the likelihoods of the `sample` for the given `data` with error `sigma` "
function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  exp(llh(data, parms, y0, sigma))
end

" given a sample, extend to all model parameters " 
function sampletoparms(sample::Vector)
  np = length(sampledinds)
  allparms = allparms(sample[1:np])
  y0 = sample[np+1:end]
  allparms, y0
end


### Self-consistency iteration / EM-Algorithm
### Old reweighting, using the non-orthogonal projection ###

" reweight the given `WeightedChain` according to its `likelihoods` "
reweight!(c::WeightedChain) = (reweight!(c.weights, c.likelihoods); c)

function reweight!(w::DenseVector, L::DenseMatrix)
  K = size(L,1)
  M = size(L,2)

  norms = L' * w

  @inbounds for k=1:K
    s = 0.
    @simd for m=1:M
      s += L[k,m] / norms[m]
    end
    w[k] = w[k] / M * s
  end
  w
end


### Maximal Likelihood for the Prior ###

" posterior for the priors evaluated at w, i.e. the probability P(data|w) "
A(w::Vector, L::Matrix) = prod(L'*w) :: Real


function dA(w::Vector, L::Matrix)
  norms = L'*w
  # TODO: check A not appearing
  A = prod(norms)
  inv = 1 ./ norms
  (L * inv) :: Vector
end

" gradient ascend of A(w) projected onto the simplex, returning the next step for stepsize h " 
function euler_A!(c::WeightedChain, h::Real)
  c.weights = euler_A(c.weights, c.likelihoods, h)
  c
end

euler_A(w, L, h) = projectsimplex!(w + dA(w, L) * h)

### Maximal posterior for the prior with entropy hyperprior

type WeightedDensity
  likelihoods::Matrix
  weights::Vector
  density::Vector
end

function WeightedDensity(L::Matrix, prior::Vector) 
  n = size(L, 1)
  weights = ones(n) / n
  density = mean(L, 2) .* prior |> vec
  DensityWeightedChain(L, weights, density) 
end

density(wd::WeightedDensity) = wd.density .* wd.weights

entropy(wd::WeightedDensity) = -dot(wd.weights, log(density(wd)))


" objective function: log of probability * entropy "
phih(wd::WeightedDensity) = log(A(wd.weights, wd.llh)) + entropy(wd)

import ForwardDiff

function euler_phih(w, pi1, L, h)
  f(w) = phih(w, pi1, L)
  g = ForwardDiff.gradient(f, w)
  projectsimplex!(w + g * h)
end

    
# TODO: use nonlinear optimizer
# to optimize phih inside the simplex
