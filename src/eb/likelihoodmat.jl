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
