type WeightedChain
  samples::Matrix
  likelihoods::Matrix
  weights::Vector
  density::Vector
end

function WeightedChain(samplings::Sampling...; sigma::Real=.1)
  samples = vcat([s.samples for s in samplings]...)
  weights = ones(size(samples, 1)) / size(samples, 1)
  lhs = likelihoods(samples, Matrix[s.model[:data].value for s in samplings], sigma)
  prior = exp(vcat([s.logprior for s in samplings]...))
  density = mean(lhs, 2) .* prior |> vec
  WeightedChain(samples, lhs, weights, density)
end 

function reweight!(c::WeightedChain) 
  reweight!(c.weights, c.likelihoods)
  c
end

function euler_A!(c::WeightedChain, h::Real) 
  c.weights = euler_A(c.weights, c.likelihoods, h)
  c
end

function euler_phih!(c::WeightedChain, h)
  c.weights = euler_phih(c.weights, c.density, c.likelihoods, h)
  c
end

### Likelihood computation

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
  Array(likelihoods)
end

# TODO: base computation on model
" compute the likelihoods of the `sample` for the given `data` with error `sigma` "
function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  lh = exp(llh(data, parms, y0, sigma))
end

" given a sample, extend to all model parameters " 
function sampletoparms(sample::Vector)
  np = length(sampledinds)
  parms = allparms(sample[1:np])
  y0 = sample[np+1:end]
  parms, y0
end


### Self-consistency iteration / EM-Algorithm
### Old reweighting, using the non-orthogonal projection ###

" reweight the given `WeightedChain` according to its `likelihoods` "
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

" gradient ascend of A(w) projected onto the simplex, returning the next step for stepsize h " 
euler_A(w, L, h) = projectsimplex!(w + dA(w, L) * h)

" posterior for the priors evaluated at w, i.e. the probability P(data|w) "
A(w::Vector, L::Matrix) = prod(L'*w) :: Real

" derivative of A "
function dA(w::Vector, L::Matrix)
  norms = L'*w
  # TODO: check A not appearing
  A = prod(norms)
  inv = 1 ./ norms
  (L * inv) :: Vector
end


### Posterior iteration for the prior with entropy hyperprior ###

import ForwardDiff

" gradient ascend of A(w) with entropy weighting "
function euler_phih(w, pi1, L, h)
  f(w) = phih(w, pi1, L)
  g = ForwardDiff.gradient(f, w)
  projectsimplex!(w + g * h)
end

" objective function: log of probability * entropy "
phih(w, pi1, L) = log(A(w, L)) + entropy(w, pi1)
function entropy(w, pi1)
  nonzero = w.!=0
  -dot(w[nonzero], (w .* pi1)[nonzero])
end
