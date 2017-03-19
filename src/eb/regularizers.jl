### marginal likelihood for w

logl(w, datas, ys, rho_std) = logl(w, likelihoodmat(datas, ys, rho_std))

logl(wx, Ldx) = sum(log(Ldx * wx))

function dlogl(w, datas, ys, rho_std)
  L = likelihoodmat(datas, ys, rho_std)
  sum(L./(L*w), 1) |> vec
end


### z-Entropy for w

# repeat the weights w to match length of zs, used for multiple z samples per y sample
function repeatweights(w, zs)
  zmult = Int(length(zs) / length(w))
  wz = repmat(w, zmult) / zmult
end

# hz(w) = \int p(z|w) log(p(z|w)) dz
# with (z,wz) ~ p(z|w) importance sampling

function hz(w::AbstractVector, ys::AbstractVector, zs::AbstractVector, rho_std)
  L = likelihoodmat(zs, ys, rho_std)
  wz = repeatweights(w, zs)
  hz(w, L, wz)
end

function hz(wx::AbstractVector, Lzx::AbstractMatrix, wz::AbstractVector=wx)
  @assert size(Lzx, 1) == length(wz)
  @assert size(Lzx, 2) == length(wx)
  @assert all(isfinite(wx))
  #@assert all(isfinite(Lzx))
  @assert all(isfinite(wz))

  rhoz = Lzx * wx           # \Int L(z|x) * pi(x) dx_j
  l = 0.
  for (r,w) in zip(rhoz, wz)
    r == 0 && continue
    l -= log(r)*w
  end
  l
end

const DHZCONT = false

dhz(w,ys,zs,rho) = DHZCONT ? dhzcont(w,ys,zs,rho) : dhzdiscr(w,ys,zs,rho)

function dhzcont(w, ys, zs, rho)
  L = likelihoodmat(zs,ys,rho)
  wz = repeatweights(w,zs)

  d=dhzloop2(L,w,wz)
  any(isnan(d)) && warn("encountered NaN in dhz() computation")
  d
end

function dhzdiscr(w,ys,zs,rho)
  nanencounters = 0

  L = likelihoodmat(zs,ys,rho) :: Matrix{Float64}
  wz = repeatweights(w,zs)
  zmult = Int(length(wz) / length(w))
  lw = length(w)

  rhoz = L*w
  rhozw = rhoz ./ wz

  d = zeros(w)
  for k = 1:length(w)
    for i = 1:length(zs)
      t = L[i,k] / rhozw[i]
      isnan(t) && (nanencounters += 1; continue)
      d[k] -= t
    end
    for m = 0:zmult-1
      t = log(rhoz[k+lw*m]) / zmult
      isnan(t) && (nanencounters += 1; continue)
      d[k] -= t
    end
  end
  nanencounters > 0 && warn("skipped over $nanencounters NaN's in dhzdiscr")
  d
end

# not yet tested, will not work on different machines
function dhzdiscr_pforsa(w,ys,zs,rho)
  nanencounters = 0

  L = likelihoodmat(zs,ys,rho) :: Matrix{Float64}
  wz = repeatweights(w,zs)
  zmult = Int(length(wz) / length(w))
  lw = length(w)

  rhoz = L*w
  rhozw = rhoz ./ wz

  #d = zeros(w)
  d = SharedArray(Float64, (length(w),))
  d[:] = 0.
  @sync @parallel for k = 1:length(w)
    for i = 1:length(zs)
      t = L[i,k] / rhozw[i]
      isnan(t) && (nanencounters += 1; continue)
      d[k] -= t
    end
    for m = 0:zmult-1
      t = log(rhoz[k+lw*m]) / zmult
      isnan(t) && (nanencounters += 1; continue)
      d[k] -= t
    end
  end
  nanencounters > 0 && warn("skipped over $nanencounters NaN's in dhzdiscr")
  d
end

# not yet tested
function dhzdiscr_pmap(w,ys,zs,rho)
  L = likelihoodmat(zs,ys,rho) :: Matrix{Float64}
  wz = repeatweights(w,zs)
  zmult = Int(length(wz) / length(w))
  lw = length(w)

  rhoz = L*w
  rhozw = rhoz ./ wz

  #d = zeros(w)
  d = SharedArray(Float64, (length(w),))
  d[:] = 0.
  pmap(1:size(L,2), L[:,k] for k in 1:size(L,2)) do k, Lk
    d = 0.
    for i = 1:length(zs)
      t = Lk[i] / rhozw[i]
      isnan(t) && continue
      d -= t
    end
    for m = 0:zmult-1
      t = log(rhoz[k+lw*m]) / zmult
      isnan(t) && continue
      d -= t
    end
    d
  end
end


# d/dw_k hz(w) = - \int p(z|w) / p(z|w) * p(z|x) * log(p(z|w)) dz - 1
# using (z,wz) ~ p(z|w) importance sampling / monte carlo integration

# note: we could also sample p(z|x) (normally distr.) directly and eval log(p(z|w)) which should require less iterations => faster for large w, wz

function dhzmatrix(L,w,wz)
  rhoz = L*w
  d = -(sum(wz .* log(rhoz) ./ rhoz .* L, 1) + 1) |> vec
end

function dhzloop(L,w,wz)
  rhoz = L*w
  d = -ones(w)

  @inbounds for x in 1:length(w)
    for z in 1:length(wz)
      d[x] -= L[z,x] / rhoz[z] * wz[z] * log(rhoz[z])
    end
  end
  d
end

function dhzloop2(L,w,wz)
  rhoz = L*w
  d = -ones(w)

  @inbounds for z in 1:length(wz)
    fact = wz[z] * log(rhoz[z]) / rhoz[z]
    @simd for x in 1:length(w)
      d[x] -= fact * L[z,x]
    end
  end
  d
end

function dhztest(n=1000,m=n)
  L  = rand(n,m)
  wz = rand(n)
  w  = rand(m)
  g = GynC.gradify(w->GynC.hz(w, L, wz),w)
  @time a=dhzmatrix(L,w,wz)
  @time b=dhzloop(L,w,wz)
  @time c=dhzloop2(L,w,wz)
  @time d=g(w)
  Base.Test.@test_approx_eq a b
  Base.Test.@test_approx_eq a c
  Base.Test.@test_approx_eq a d
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

const hz_simdays = 31 # does this belong here?
const rho_e = MvNormal(model_measerrors)

phi(x::Vector{Float64}) = forwardsol(x, 0:hz_simdays-1)[:, measuredinds] :: Matrix{Float64}
phi(x::Matrix) = pmap(phi, [x[i,:] for i in 1:size(x, 1)]) |> Vector{Matrix{Float64}}

zt(zs::Vector) = map(z->z+rand(rho_e, hz_simdays)', zs)
