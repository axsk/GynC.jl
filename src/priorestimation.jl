##### Data structures #####

# WeightedChain represents a weighted sampling of a distribution
type WeightedChain
  # sample coordinates in each row
  samples::Matrix

  # likelihoods for the different datas in each column 
  likelihoods::Matrix

  # the corresponding weights 
  weights::Vector

  # the density of the initial sampling, for e.g. entropy calculation
  density::Vector
end

# construct the WeightedChain corresponding to the concatenated samples and respective data
function WeightedChain(samplings::Sampling...; sigma::Real=.1)
  samples = vcat([s.samples for s in samplings]...)
  prior   = vcat(([exp(s.logprior) for s in samplings]...))
  lhs     = likelihoods(samples, Matrix[s.model[:data].value for s in samplings], sigma)
  WeightedChain(samples, lhs, prior)
end

# construct the WeightedChaind corresponding to the given likelihoods and priors
function WeightedChain(samples::Matrix, lhs::Matrix, prior::Vector)
  weights = ones(size(samples, 1)) / size(samples, 1)
  lhs     = lhs ./ sum(lhs, 1) # normalize for stability
  density = mean(lhs, 2) .* prior |> vec
  density = density / sum(density) # normalize for entropy
  WeightedChain(samples, lhs, weights, density)
end

  

### wrappers ###

emiteration!(c::WeightedChain) = (emiteration!(c.weights, c.likelihoods); c)
euler_A!(c::WeightedChain, h::Real) = (c.weights = euler_A(c.weights, c.likelihoods, h); c)
euler_phih!(c::WeightedChain, h) = (c.weights = euler_phih(c.weights, c.density, c.likelihoods, h); c)

### Likelihood computation
## TODO: move this to model.jl

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


##### Algorithms #####

### Self-consistency iteration / EM-Algorithm ###

# iterative application of the prior estimation step
# which is equivalent to the expectation-maximization of the prior π

#                L(k|m) * w(k) 
# w(k) <-  ∑ ---------------------
#            M * ∑ L(k'|m) * w(k')

@deprecate reweight!(w, L) emiteration!(w, L)
function emiteration!(w::DenseVector, L::DenseMatrix)
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


### Hyperlikelihood optimization ###

# We can instead try to immediately optimize the hyperlikelihood
# A(π) := L(z|π) = ∏ L(z_m|π) = ∏ ∫ L(z_m|p,π) * P(p|π)
# A(w) := ∏ L_kj * w_k
A(w::Vector, L::Matrix) = prod(L'*w) :: Real

# with derivative
# dA(w)_k = A * L_kj / (L_ij * w_i)
function dA(w::Vector, L::Matrix)
  norms = L'*w
  A = prod(norms)
  inv = 1 ./ norms
  A * (L * inv) :: Vector
end

# a direct approach is to 
# project euler steps onto the simplex

euler_A(w, L, h) = projectsimplex!(w + dA(w, L) * h)


### Entropy weighted Hyperposterior optimization ###

# since this results merely in a maximum likelihood estimate, which in application often is irregular/unregular? 
# one may try to regularize this with an entropy based prior
# P(π) = h(π) = - ∫ π(x) * log(π(x)) dx

function entropy(weights, density)
  h = 0.
  for i in eachindex(weights)
    p = weights[i] * density[i]
    p == 0 && continue
    h += weights[i] * log(p)
  end
  -h
end

# and then optimize the resulting 
# posterior P(π|z) ~=  P(z|π) * h(π)

## unser objective hier is jetzt eher log(A(π) * e^h(π)), warum?
phih(w, pi1, L) = log(A(w, L)) + entropy(w, pi1)

# with
# the euler-projection iteration 

function euler_phih(w, pi1, L, h)
  f(w) = phih(w, pi1, L)
  g = ForwardDiff.gradient(f, w)
  projectsimplex!(w + g * h)
end

# to obtain the MAP of the Hyperposterior.
