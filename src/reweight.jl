### WeightedChain

type WeightedChain
  chain::AbstractMatrix
  weights::AbstractVector
  likelihoods::AbstractMatrix
end

function WeightedChain(chain::AbstractMatrix, data::Vector{Matrix}, sigma::Real)
  WeightedChain(chain, ones(size(chain, 1)), likelihoods(chain, data, sigma))
end

function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  exp(llh(data, parms, y0, sigma))
end

function sampletoparms(sample::Vector)
  parms = copy(mleparms)
  parms[sampleparms] = sample[1:length(sampleparms)]
  y0 = sample[length(sampleparms)+1:end]
  parms, y0
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

function reweight!(c::WeightedChain)
  w = c.weights
  L = c.likelihoods
  K = size(L,1)
  M = size(L,2)
  norm = [sum([w[k] * L[k,m] for k=1:K]) for m=1:M]
  for k=1:K
    w[k] = w[k] / M * sum([L[k,m] / norm[m] for m=1:M])
  end
  w
end
