phi(x::Vector) = forwardsol(x, 1:31)[:, measuredinds]
phi(x::Matrix) = [phi(x[i,:] |> vec) for i in 1:size(x, 1)]

function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w
  rhoz = rhoz / sum(rhoz)
  -dot(log(rhoz), w)
end

"construct the objective function w -> L + Hz"
function hzobj(x::Matrix{Float64}, datas::Vector{Matrix{Float64}})
  zsim = phi(x)
  zerr = addmeaserror(zsim)

  L_data_zs = llh_measerror(datas, zsim)
  L_zerr_zs = llh_measerror(zerr, zsim)

  w -> sum(L_data_zs * w) / length(datas) + Hz(w, L_zerr_zs)
end

const rho_e = MvNormal(model_measerrors)
addmeaserror(zs::Vector) = map(z->z+rand(rho_e, size(zs[1],1))', zs)

# likelihood of measurments
function llh_measerror(z1::Vector, z2::Vector)
  [llh_measerror(a - b) for a in z1, b in z2]
end
