
### likelihood matrix compilation

# compute the likelihoods of measuring zs given ys, return the cached matrix
@deprecate likelihoodmat(zs, ys, rho_std) likelihoodmat(zs, ys, MvNormal(2, rho_std))

@memoize Dict function likelihoodmat(zs, ys, d::Distribution)
  print("computing likelihood matrix ($(length(zs))x$(length(ys)))")
  #@time L = [pdf(d, z-y) for z in zs, y in ys]
  #t=@elapsed (L=likelihoodmat_par(zs,ys,d))
  t=@elapsed (L=likelihoodmat_nanfast(zs,ys,d))
  println(" ($t seconds)")
  L::Matrix{Float64}
end

# parallelize over columns
function likelihoodmat_par(zs, ys, d::Distribution)
  x = SharedArray(Float64, (length(zs), length(ys)))
  @sync @parallel for j = 1:length(ys)
    for i = 1:length(zs)
      @inbounds x[i,j] = pdf(d, zs[i]-ys[j])
    end
  end
  Array(x)
end

function likelihoodmat_nanfast(xs,ys,d)
  A = hcat([vec(x ./ d.sigmas) for x in xs]...)
  B = hcat([vec(y ./ d.sigmas) for y in ys]...)
  exp(-sqeucdist_nan(A,B))
end


function sqeucdist_nan(A,B)
  An = isnan(A)
  Bn = isnan(B)

  A[An] = 0
  B[Bn] = 0
  sa2 = sumabs2(A,1)
  sb2 = sumabs2(B,1)

  ca = zeros(size(A,2), size(B,2))

  for ind in zip(findn(An)...)
    k,i = ind
    ca[i,:] += abs2(B[k, :])
  end

  for ind in zip(findn(Bn)...)
    k,j = ind
    ca[:,j] += abs2(A[k, :])
  end

  r = A'*B

  @inbounds for j = 1:size(B,2)
    for i = 1:size(A,2)
      v = sa2[i] + sb2[j] - 2 * r[i,j] # binomial formula
      v = v - ca[i,j] # correct the terms which shouldnt count due to nan
      r[i,j] = v
    end
  end
  r
end


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

function hz(w::Vector, ys::Vector, zs::Vector, rho_std)
  L = likelihoodmat(zs, ys, rho_std)
  wz = repeatweights(w, zs)
  hz(w, L, wz)
end

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
