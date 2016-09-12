# given X-samples `x`, the forwardd map `phi`:X->Z, and the error density `err`, compute the pairwise likelihood matrix Lz, corresponding to L(x|z)

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


phi(x::Vector) = forwardsol(x, 1:31)
phi(x::Matrix) = [phi(x[i,:]) for i in 1:size(x, 1)]

function obj(x::Matrix, w::Vector, datas::Vector, sigma::Real)
  z = phi(x)
  L = [llh(z, data, sigma) for z in z, data in datas]
  err(z1, z2) = exp(llh(z1, z2, sigma))
  A(w,L) + Hz(w, Lz(z, err))
end

# log L(z|data), i.e. the measurement error, up to a constant
llh(z, data, sigma) = znorm(z, data) / (2*sigma^2)
