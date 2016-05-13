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

function GynCChain(samplings::Sampling...; sigma::Real=.1)
  x = vcat([s.samples for s in samplings]...)
  w = ones(size(x, 1))
  l = likelihoods(x, Matrix[s.model[:data].value for s in samplings], sigma)
  GynCChain(x,w,l)
end 

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
  parms = allparms(sample[1:np])
  y0 = sample[np+1:end]
  parms, y0
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


### Maximal Likelihood / Posterior for the Prior ###

" posterior for the priors evaluated at w, 
i.e. the probability P(data|w) "
A(w::Vector, L::Matrix) = prod(L'*w) :: Real

function dA(w::Vector, L::Matrix)
  norms = L'*w
  # TODO: check A not appearing
  A = prod(norms)
  inv = 1 ./ norms
  (L * inv) :: Vector
end

" compute the entropy of `pi_k` given `pi_1` and `w_k` "
entropy(pi_1, w) = -dot(w, log(pi_1.*w))

" objective function: log of probability * entropy "
phih(w, pi_1, L) = log(A(w, L)) + entropy(pi_1, w)

import ForwardDiff

function euler_phih(w, pi_1, L, h)
  f(w) = phih(w, pi_1, L)
  g = ForwardDiff.gradient(f, w)
  projectsimplex!(w + g * h)
end

    
# TODO: use nonlinear optimizer
# to optimize phih inside the simplex

" gradient ascend of A(w) projected onto the simplex, returning the next step for stepsize h " 
function euler_A!(c::WeightedChain, h::Real)
  c.weights = euler_A(c.weights, c.likelihoods, h)
  c
end

euler_A(w, L, h) = projectsimplex!(w + dA(w, L) * h)
