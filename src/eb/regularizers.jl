# compute the likelihoods of measuring zs given ys, return the cached matrix
@memoize function likelihoodmat(zs, ys, rho_std)
  info("computing likelihood matrix")
  N = MvNormal(2, rho_std)
  L = [pdf(N, z-y) for z in zs, y in ys]
end


### marginal likelihood for w

logLw(wx, Ldx) = sum(log(Ldx * wx))

logLw(w, xs, datas, rho_std) = logLw(w, likelihoodmat(datas, xs, rho_std))


### z-Entropy for w

function Hz(w::Vector, ys::Vector, zs::Vector, rho_std::Real)
  L = likelihoodmat(zs, ys, rho_std)
  zmult = Int(length(zs) / length(ys))
  wz = repmat(w, zmult) / zmult
  Hz(w, L, wz)
end

function Hz(wx::Vector, Lzx::Matrix, wz::Vector=wx)
  @assert size(Lzx, 1) == length(wz)
  @assert size(Lzx, 2) == length(wx)

  rhoz = Lzx * wx           # \Int L(z|x) * pi(x) dx_j
  l = 0
  for (r,w) in zip(rhoz, wz)
    r == 0 && continue
    l -= log(r)*w
  end
  l
end


### hkl

# compute the integral \int p(z|pi) D_KL(pi(x), p(x|pi)) dz
# L[l,k] = L(zl | xk), zl = phi(xk)+el
function HKL(w, L, wz=w)
  #@assert length(w) == size(Ldz, 2)
  evidences = L * w
  h = 0.
  for k = 1:size(L, 2)
    for l = 1:size(L, 1)
      h += wz[l] * L[l, k] * w[k] / evidences[l] * log( L[l,k] / evidences[l])
    end
  end
  h / size(L, 1) / size(L, 2)
end


# old gync compability layer

const hz_simdays = 31
const rho_e = MvNormal(model_measerrors)

phi(x::Vector{Float64}) = forwardsol(x, 0:hz_simdays-1)[:, measuredinds] :: Matrix{Float64}
phi(x::Matrix) = pmap(phi, [x[i,:] for i in 1:size(x, 1)]) |> Vector{Matrix{Float64}}

zt(zs::Vector) = map(z->z+rand(rho_e, hz_simdays)', zs)

