# given X-samples `x`, the forwardd map `phi`:X->Z, and the error density `err`, compute the pairwise likelihood matrix Lz, corresponding to L(x|z)
function Lz(x, phi, err)
  z = map(phi, xs)
  n = length(z)
  L = Array{Float64}(n,n)
  for i=1:n
    for j=i:n
      L[i,j] = L[j,i] = err(z[i], z[j])
    end
  end
  L::Matrix{Float64}
end

Hz(pi, phi, err) = Hz(pi, Lz(pi, phi, err))

# compute the prior predictive distribution entropy
# w - the weights of the samples
# Lz - the z likelihood matrix

function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), w)
end


#= old 

# compute the marginal likelihood / evidence entropy
# compute the entropy of the prior predictive distribution (of z given pi)
function Hz(pi, L, phi)
  @assert sum(weights(pi)) == 1
  zs = map(phi, pi)
  rhoz = [dot([L(z,x) for x in samples(pi)], weights(pi)) for z in zs]
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), weights(pi))
end
=#
