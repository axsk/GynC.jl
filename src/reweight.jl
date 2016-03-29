### WeightedChain

type WeightedChain
  chain::AbstractMatrix
  weights::AbstractVector
  likelihoods::AbstractMatrix
end

WeightedChain(c::Vector, w, l) = WeightedChain(reshape(c,length(c),1), w, l)

function WeightedChain(chain::Matrix, data::Vector{Matrix}, sigma::Real)
  WeightedChain(chain, ones(size(chain, 1)), likelihoods(chain, data, sigma))
end

function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  exp(llh(data, parms, y0, sigma))
end

function sampletoparms(sample::Vector)
  np = length(sampledinds)
  allparms = allparms(sample[1:np])
  y0 = sample[np+1:end]
  allparms, y0
end

""" compute the likelihood matrix for given chains, data, sigma) """
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

" reweight the given `WeightedChain` and return a `Dict` with the iterations given in `range` "
function reweight(c::WeightedChain, range)
  res = Dict{Int, WeightedChain}()
  pi = deepcopy(c)

  for i in 0:maximum(range)
    in(i, range) && push!(res, i=>deepcopy(pi))
    reweight!(pi)
  end
  res
end


" reweight the given `WeightedChain` according to its `likelihoods` "
reweight!(c::WeightedChain) = reweight!(c.weights, c.likelihoods)

function reweight!(w::DenseVector, L::DenseMatrix)
  K = size(L,1)
  M = size(L,2)
  norm = Array{Float64}(M)
  @inbounds for m=1:M
    s = 0.
    @simd for k=1:K
      s += w[k] * L[k,m]
    end
    norm[m] = s
  end

  @inbounds for k=1:K
    s = 0.
    @simd for m=1:M
      s += L[k,m] / norm[m]
    end
    w[k] = w[k] / M * s
  end
  w
end

### simplex projection ###

# c.f. https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat_simplexproj.pdf"

" project the vector y onto the unit simplex minimizing the euclidean distance " 
projectsimplex(y)  = projectsimplex!(copy(y))

" in-place version of `projectsimplex` "
projectsimplex!(y) = projectsimplex_heap!(y)

" heap implementation (algorithm 2) "
function projectsimplex_heap!{T <: Real}(y::Array{T, 1})
  heap = heapify(y, Base.Order.Reverse)
  cumsum = zero(T)
  t = zero(T)
  for k in 1:length(y)
    uk = heappop!(heap, Base.Order.Reverse)
    cumsum += uk
    normalized = (cumsum - one(T)) / k  
    normalized >= uk && break
    t = normalized
  end
  for i in 1:length(y)
    y[i] = max(y[i] - t, zero(T))
  end
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
end
