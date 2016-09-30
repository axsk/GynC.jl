### simplex projection algorithms ###
# c.f. https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat_simplexproj.pdf"

" project the vector y onto the unit simplex minimizing the euclidean distance " 
projectsimplex(y)  = projectsimplex!(copy(y))

" in-place version of `projectsimplex` "
projectsimplex!(y) = projectsimplex_heap!(y)

" heap implementation (algorithm 2) "
function projectsimplex_heap!{T <: Real}(y::Array{T, 1})
  heap = Collections.heapify(y, Base.Order.Reverse)
  cumsum = zero(T)
  t = zero(T)
  for k in 1:length(y)
    uk = Collections.heappop!(heap, Base.Order.Reverse)
    cumsum += uk
    normalized = (cumsum - one(T)) / k  
    normalized >= uk && break
    t = normalized
  end
  for i in 1:length(y)
    y[i] = max(y[i] - t, zero(T))
  end
  y
end

" sort implementation (algorithm 1), non-allocating when provided `temp` "
function projectsimplex_sort!{T <: Real}(y::Array{T, 1}, temp=similar(y))
  copy!(temp, y)
  sort!(temp, rev=true)
  cumsum = zero(T)
  t = zero(T)
  for k in 1:length(y)
    uk = temp[k]
    cumsum += uk
    normalized = (cumsum - one(T)) / k
    normalized >= uk && break
    t = normalized
  end
  for i in 1:length(y)
    y[i] = max(y[i] - t, zero(T))
  end
  y
end

" given `w` and `grad`, compute `Psi_S(w) := w + S_w * grad`
S_w is constructed in a manner to enforce that the result lies on the unit simplex "
function psi_S(w, gradient)
  w + w.* gradient - w*dot(w, gradient)
end
