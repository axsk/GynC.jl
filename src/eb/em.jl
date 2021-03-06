### wrappers ###

using Iterators

emiterations(w0, L, niter) = collect(take(iterate(w->emiteration(w,L), w0), niter))

emiteration!(c::WeightedChain) = (emiteration!(c.weights, c.likelihoods); c)
euler_A!(c::WeightedChain, h::Real) = (c.weights = euler_A(c.weights, c.likelihoods, h); c)
euler_phih!(c::WeightedChain, h) = (c.weights = euler_phih(c.weights, density(c), c.likelihoods, h); c)


##### Algorithms #####

### Self-consistency iteration / EM-Algorithm ###

# iterative application of the prior estimation step
# which is equivalent to the expectation-maximization of the prior π

#                L(k|m) * w(k) 
# w(k) <-  ∑ ---------------------
#            M * ∑ L(k'|m) * w(k')

emiteration(w, L) = emiteration!(copy(w), L)

@deprecate reweight!(w, L) emiteration!(w, L)

function emiteration!(w::DenseVector, L::DenseMatrix)
  K = size(L,1)
  M = size(L,2)

  norms = L' * w

  @inbounds for k=1:K
    s = 0.
    for m=1:M
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
    density[i] == 0 && continue
    h += weights[i] * log(density[i]) ### ISNT THIS WRONG? should be density * weights
  end
  -h
end

# and then optimize the resulting 
# posterior P(π|z) ~=  P(z|π) * h(π)

phih(w, pi1, L) = log(A(w, L)) + entropy(w, pi1)

# with
# the euler-projection iteration 

function euler_phih(w, pi1, L, h)
  f(w) = phih(w, pi1, L)
  g = ForwardDiff.gradient(f, w)
  projectsimplex!(w + g * h)
end

# to obtain the MAP of the Hyperposterior.
