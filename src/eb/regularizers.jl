###  high-level api

type LikelihoodModel
  xs::Vector
  ys::Vector
  zs::Vector
  datas::Vector
  measerr::Distribution
end

function mple(m::LikelihoodModel, w0, niter, reg, h)
  c1 = 1/100
  c2 = 1/1000
  ndata = length(m.datas)
  hauto = h/((1-reg)*(ndata/c1) + reg/c2)
  gradientascent(mple_obj(m, reg), w0, niter, hauto, autodiff=false)
end

function mple_obj(m::LikelihoodModel, reg)
  w -> reg*dhz(w, m.ys, m.zs, m.measerr) + (1-reg) * dlogl(w, m.datas, m.ys, m.measerr)
end

function em(m::LikelihoodModel, w0, niter)
  L = likelihoodmat(m.ys, m.datas, m.measerr)
  emiterations(w0, L, niter)
end

hz(m::LikelihoodModel, w) = hz(w, m.ys, m.zs, m.measerr)
logl(m::LikelihoodModel, w) = logLw(w, m.ys, m.datas, m.measerr)


### internal calculations


# compute the likelihoods of measuring zs given ys, return the cached matrix
@deprecate likelihoodmat(zs, ys, rho_std) likelihoodmat(zs, ys, MvNormal(2, rho_std))

@memoize function likelihoodmat(zs, ys, merr::Distribution)
  info("computing likelihood matrix ($(length(zs))x$(length(ys)))")
  #@time L = [pdf(merr, z-y) for z in zs, y in ys]
  likelihoodmat_par(zs,ys,merr)
end

# parallelize over columns
function likelihoodmat_par(zs, ys, merr::Distribution)
  x = SharedArray(Float64, (length(zs), length(ys)))
  @sync @parallel for j = 1:length(ys)
    for i = 1:length(zs)
      @inbounds x[i,j] = pdf(merr, zs[i]-ys[j])
    end
  end
  Array(x)
end


### marginal likelihood for w

logLw(w, ys, datas, rho_std) = logLw(w, likelihoodmat(datas, ys, rho_std))

logLw(wx, Ldx) = sum(log(Ldx * wx))

function dlogl(w, datas, ys, rho_std)
  L = likelihoodmat(datas, ys, rho_std)
  sum(L./(L*w), 1) |> vec
end

### z-Entropy for w

function hz(w::Vector, ys::Vector, zs::Vector, rho_std)
  L = likelihoodmat(zs, ys, rho_std)
  zmult = Int(length(zs) / length(ys))
  wz = repmat(w, zmult) / zmult
  hz(w, L, wz)
end

# hz(w) = \int p(z|w) log(p(z|w)) dz
# with (z,wz) ~ p(z|w) importance sampling

function hz(wx::Vector, Lzx::Matrix, wz::Vector=wx)
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

function repeatweights(w, zs)
  zmult = Int(length(zs) / length(w))
  wz = repmat(w, zmult) / zmult
end

function dhz(w, ys, zs, rho)
  L = likelihoodmat(zs,ys,rho)
  wz = repeatweights(w,zs)

  dhzloop2(L,w,wz)
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
  d = zeros(w)

  @inbounds for x in 1:length(w)
    for z in 1:length(wz)
      d[x] -= L[z,x] / rhoz[z] * wz[z] * log(rhoz[z])
    end
  end
  d-1
end

function dhzloop2(L,w,wz)
  rhoz = L*w
  d = fill!(similar(w), -1.0)

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

const hz_simdays = 31
const rho_e = MvNormal(model_measerrors)

phi(x::Vector{Float64}) = forwardsol(x, 0:hz_simdays-1)[:, measuredinds] :: Matrix{Float64}
phi(x::Matrix) = pmap(phi, [x[i,:] for i in 1:size(x, 1)]) |> Vector{Matrix{Float64}}

zt(zs::Vector) = map(z->z+rand(rho_e, hz_simdays)', zs)
