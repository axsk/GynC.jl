# given X-samples `x`, the forwardd map `phi`:X->Z, and the error density `err`, compute the pairwise likelihood matrix Lz, corresponding to L(x|z)

Lz(x, phi, err) = Lz(map(phi, x), err)

function Lz(z::Vector, err::Function)
  z = map(phi, xs)
  n = length(z)
  L = Array{Float64}(n,n)
  for i=1:n
    for j=i:n
      L[i,j] = L[j,i] = err(z[i], z[j])
    end
  end
  L
end

# compute the prior predictive distribution entropy
# w - the weights of the samples
# Lz - the z likelihood matrix

Hz(pi, phi, err) = Hz(pi, Lz(pi, phi, err))

function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), w)
end

# what we need
samples of different persons
merge them
compute likelihoods for each person
co

objective
 likelihood
  personal likelihoods
   solutions
 entropy
  likelihoods of x generating z
   solutions
end

phi(x) = forwardsol(x, 1:31)

function obj(x,w, datas)
  z = map(phi, x)

  L = map(llhs, datas)
  A(w,L) + Hz(w, Lz(z, l2))
end
  
