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

" compute the likelihoods of the `sample` for the given `data` with error `sigma` "
function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  lh = exp(llh(data, parms, y0, sigma))
end
