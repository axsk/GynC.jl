function hzobj(samples::Matrix{Float64}, datas::Vector{Matrix{Float64}})
  zs = phi(samples)
  Ld = Lzz(datas, zs)     # L(d|z)
  Lz = Lzz(zt(zs), zs)    # L(zt|z), zt=z+e
  w -> penalized_llh(w, Ld, Lz)
end

penalized_llh(w, Ld, Lz) = logLw(w, Ld) + Hz(w, Lz)

# marginal likelihood for w
logLw(w, Ld) = sum(log(Ld * w)) / size(Ld, 1)

# z-entropy
function Hz(w::Vector, Lz::Matrix)
  rhoz = Lz * w           # \Int L(z|x) * pi(x) dx_j
  rhoz = rhoz / sum(rhoz) # \Int rhoz(z) dz = 1
  -dot(log(rhoz), w)      # -\Int log(rhoz(z)) * rhoz(z) dz
end

###

const hz_simdays = 31
const rho_e = MvNormal(model_measerrors)

phi(x::Vector) = forwardsol(x, 0:hz_simdays-1)[:, measuredinds]
phi(x::Matrix) = [phi(x[i,:]) for i in 1:size(x, 1)]

zt(zs::Vector) = map(z->z+rand(rho_e, hz_simdays)', zs)

Lzz(a, b) = [exp(llh_measerror(zi - zj)) for zi in a, zj in b]
