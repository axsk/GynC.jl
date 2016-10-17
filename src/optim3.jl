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
  h / size(Lzz, 1) / size(Lzz, 2)
end
