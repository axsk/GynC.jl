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
