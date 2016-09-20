"compute the prior predictive distribution entropy
 w - the weights of the samples
 Lz - the z likelihood matrix (see above)"
function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), w)
end

"construct the objective function w -> L + Hz"
function hzobj(x::Matrix{Float64}, datas::Vector{Matrix{Float64}})
  zsim = phi(x)
  zerr = addmeaserror(zsim)

  L_data_zs = L(datas, zsim)
  L_zerr_zs = L(zerr, zsim)
  function o(w::Vector)
    prod(L_data_zs * w) + Hz(w, L_zerr_zs)
  end
end



addmeaserror(zs::Vector) = map(z->z+rand(rho_e, size(zs[1],1))', zs)

# likelihood of measurments
L(z1::Vector, z2::Vector)   = Float64[rho_e_m(a - b) for a in z1, b in z2]


### Model specific ###

# forward solution
phi(x::Vector) = forwardsol(x, 1:31)[:, measuredinds]
phi(x::Matrix) = [phi(x[i,:] |> vec) for i in 1:size(x, 1)]

const rho_e = MvNormal([120,10,400,15] * 0.1)

function rho_e_m(x::Matrix{Float64})
  p = 0.
  for i = 1:size(x, 1)
    row = x[i,:] |> vec
    any(isnan(row)) && continue
    p += logpdf(rho_e, row)
  end
  exp(p)
end
