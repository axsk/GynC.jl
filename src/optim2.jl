function hzobj(samples::Matrix{Float64}, datas::Vector{Matrix{Float64}})
  zs = phi(samples)
  Ld = Lzz(datas, zs)     # L(d|z)
  Lz = Lzz(zt(zs), zs)    # L(zt|z), zt=z+e
  w -> penalized_llh(w, Ld, Lz)
end

penalized_llh(w, Ld, Lz) = logLw(w, Ld) + Hz(w, Lz)

# marginal likelihood for w
logLw(wx, Ldx) = sum(log(Ldx * wx))

# z-entropy
function Hz(wx::Vector, Lzx::Matrix, wz::Vector=wx)
  @assert size(Lzx, 1) == length(wz)
  @assert size(Lzx, 2) == length(wx)

  rhoz = Lzx * wx           # \Int L(z|x) * pi(x) dx_j
  rhoz = rhoz/sum(rhoz)
  l = 0
  for (r,w) in zip(rhoz, wz)
    r == 0 && continue
    l -= log(r)*w
  end
  l
end

log0(x) = [x==0 ? 0 : log(x) for x in x]

###

const hz_simdays = 31
const rho_e = MvNormal(model_measerrors)

phi(x::Vector{Float64}) = forwardsol(x, 0:hz_simdays-1)[:, measuredinds] :: Matrix{Float64}
phi(x::Matrix) = pmap(phi, [x[i,:] for i in 1:size(x, 1)]) |> Vector{Matrix{Float64}}

zt(zs::Vector) = map(z->z+rand(rho_e, hz_simdays)', zs)

Lzz(a, b) = [exp(llh_measerror(zi - zj)) for zi in a, zj in b]
