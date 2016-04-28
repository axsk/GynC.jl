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

  #= performing slower?
  norm = Array{Float64}(M)
  @inbounds for m=1:M
    s = 0.
    @simd for k=1:K
      s += w[k] * L[k,m]
    end
    norm[m] = s # rho(z_m)
  end
  =#

  norm = L' * w

  @inbounds for k=1:K
    s = 0.
    @simd for m=1:M
      s += L[k,m] / norm[m]
    end
    w[k] = w[k] / M * s
  end
  w
end


### Maximal Likelihood / Posterior for the Prior ###
# TODO: use nonlinear optimizer

" posterior for the priors evaluated at w "
A(w::Vector, L::Matrix) = prod(L'*w) :: Real

function dA(w::Vector, L::Matrix)
  norms = L'*w
  A = prod(norms)
  inv = 1 ./ norms
  (L * inv) :: Vector
end

" - /int pi(x) log(pi(x)) dx "
function H(pi)
  # TODO: make this integral work for non-uniform grid for the pi
  -dot(x, log(x))
end

#function PhiH = log(A) - H
    

" gradient ascend of A(w) projected onto the simplex, returning the next step for stepsize h " 
function gradient_simplex!(c::WeightedChain, h::Real)
  c.weights = gradient_simplex(c.weights, c.likelihoods, h)
  c
end

gradient_simplex(w,L,h) = projectsimplex!(w + dA(w, L) * h) 


### simplex projection algorithms ###

# c.f. https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat_simplexproj.pdf"

" project the vector y onto the unit simplex minimizing the euclidean distance " 
projectsimplex(y)  = projectsimplex!(copy(y))

" in-place version of `projectsimplex` "
projectsimplex!(y) = projectsimplex_heap!(y)

" heap implementation (algorithm 2) "
function projectsimplex_heap!{T <: Real}(y::Array{T, 1})
  heap = Collections.heapify(y, Base.Order.Reverse)
  cumsum = zero(T)
  t = zero(T)
  for k in 1:length(y)
    uk = Collections.heappop!(heap, Base.Order.Reverse)
    cumsum += uk
    normalized = (cumsum - one(T)) / k  
    normalized >= uk && break
    t = normalized
  end
  for i in 1:length(y)
    y[i] = max(y[i] - t, zero(T))
  end
  y
end

" sort implementation (algorithm 1), non-allocating when provided `temp` "
function projectsimplex_sort!{T <: Real}(y::Array{T, 1}, temp=similar(y))
  copy!(temp, y)
  sort!(temp, rev=true)
  cumsum = zero(T)
  t = zero(T)
  for k in 1:length(y)
    uk = temp[k]
    cumsum += uk
    normalized = (cumsum - one(T)) / k
    normalized >= uk && break
    t = normalized
  end
  for i in 1:length(y)
    y[i] = max(y[i] - t, zero(T))
  end
  y
end
