using LineSearches

#using GynC

#f=GynC.mple_obj(m, .5)
#df=GynC.dmple_obj(m .5)

const SIMPLEXMINDIST = 1e-12

" do a linesearch for given f, df from x in direction of the gradient orthogonally projected onto the simplex "
function simplexlinesearch(f,df,x, alpha0 = 1, relboundarystep=0)
  # start in simplex
  x = proj(x) 

  # restrict f to simplex evaluations
  pf(x) = f(proj(x))

  # restrict df to simplex interior evaluations
  pdf(x) = df(projsimplexint(x))

  # initialisation
  lsdf = LineSearches.DifferentiableFunction(pf,(x,g)->(g[:]=pdf(x)))

  # allocation and initial evaluation
  p = similar(x)
  fx = lsdf.fg!(x,p)

  # search direction and slope
  phi = -projsimplextangent(p)
  @show dphi = dot(phi, p)

  # when on/near border ignore directions outwards
  for i in eachindex(phi)
    if phi[i] < 0 && x[i] <= SIMPLEXMINDIST
      phi[i] = 0
    end
  end

  # maximally move to the border: x + a phi = 0
  if relboundarystep > 0
    @show alpha0 = min(minimum(-x[i] / phi[i] for i in 1:length(x) if phi[i] < 0), alpha0) * relboundarystep
  end

  # initial tape
  lsr = LineSearchResults(eltype(x))
  push!(lsr, 0.0, fx, dphi)

  tx = copy(x)
  @show alpha, _, _ = backtracking!(lsdf, x, phi, tx, p, lsr, alpha0)
  @show lsr

  proj(x + alpha * phi)
end

function projsimplexint(x)
  y = GynC.projectsimplex(x)
  i0 = find(x->x==0, y)
  @show length(i0)
  y[i0] .+= SIMPLEXMINDIST
  y / sum(y)
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

