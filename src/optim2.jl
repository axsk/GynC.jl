"compute the pairwise likelihood matrix Lz, with Lz_ij = L(x_i|z_j)"
function Lz(z::Vector, err::Function)
  n = length(z)
  L = Array{Float64}(n,n)
  for i=1:n
    for j=i:n
      L[i,j] = L[j,i] = err(z[i], z[j])
    end
  end
  L
end

"compute the prior predictive distribution entropy
 w - the weights of the samples
 Lz - the z likelihood matrix (see above)"
function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), w)
end

"construct the objective function w -> L + Hz"
function hzobj(x::Matrix, datas::Vector, err::Function)
  z = phi(x)
  L = [err(z, data) for z in z, data in datas]
  Lz = Lz(z, err)
  function o(w)
    A(w,L) + Hz(w, Lz)
  end
end

hzobj(x::Matrix, datas::Vector, sigma::Real) = hzobj(x, datas, (a, b) -> rho_e(a, b, sigma))

### Model specific ###

# forward solution
phi(x::Vector) = forwardsol(x, 1:31)[:, measuredinds]
phi(x::Matrix) = [phi(x[i,:] |> vec) for i in 1:size(x, 1)]

# log L(z|data), i.e. the measurement error, up to normalization
rho_e(z1, z2, sigma) = exp(-znorm(z1, z2) / (2*sigma^2))
