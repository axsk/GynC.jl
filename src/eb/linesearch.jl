using LineSearches

#using GynC

#f=GynC.mple_obj(m, .5)
#df=GynC.dmple_obj(m .5)

" do a linesearch for given f, df from x in direction of the gradient orthogonally projected onto the simplex "
function simplexlinesearch(f,df,x, alpha0 = 1, relboundarystep=0)
  # initialisation
  lsdf = LineSearches.DifferentiableFunction(f,(x,g)->(g[:]=df(x)))

  # allocation and initial evaluation
  tx = copy(x)
  p = similar(x)
  fx = lsdf.fg!(x,p)

  # search direction and slope
  phi = -projsimplextangent(p)
  #phi = -p
  @show dphi = dot(phi, p)

  # maximally move to the border: x + a phi = 0
  if relboundarystep > 0
    @show alpha0 = min(minimum(-x[i] / phi[i] for i in 1:length(x) if phi[i] <= 0), alpha0) * relboundarystep
  end

  # initial tape
  lsr = LineSearchResults(eltype(x))
  push!(lsr, 0.0, fx, dphi)

  @show alpha, _, _ = backtracking!(lsdf, x, phi, tx, p, lsr, alpha0, false, 1e-4)
  @show lsr

  x + alpha * phi
end

function projsimplextangent(x)
  x - sum(x) / length(x)
end

proj(x) = GynC.projectsimplex(x)

function linesearch(m; n=3, reg=0.9)
  local x = fill(1/length(m.xs), length(m.xs))

  x=Base.normalize(rand(length(m.xs)),1)
  f=GynC.mple_obj(m, reg) # maximization objective
  df=GynC.dmple_obj(m, reg)

  pfm(x) = - f(proj(x)) # proj f to minimize

  dfm(x) = - df(x)

  @show pfm(x)

  for i = 1:n
    x = simplexlinesearch(pfm, dfm, x)
    nz = sum(x.<0)
    no = sum(x.>1)
    s  = sum(x)
    @show nz, no, s
    @show pfm(x)
    #@show x[1], proj(x)[1]
    #x = proj(x)
    #@show x[1]
    #@assert proj(x) == x
    #@show sum(x.==0)
    #@show sum(proj(x).==0)

    #show pfm(x) + f(x)
    #show pfm(x) + f(proj(x))
    #@show sum(x .<= 1e-7)
    #@show sort(x)[1:5]
    #@show sort(x)[end-4:end]
  end

  #@show pfm(x)
  x
end

