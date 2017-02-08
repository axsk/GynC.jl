using LineSearches

#using GynC

#f=GynC.mple_obj(m, .5)
#df=GynC.dmple_obj(m .5)

" do a linesearch for given f, df from x in direction of the gradient orthogonally projected onto the simplex "
function linesearch(f,df,x, alpha0=1.)
  # initialisation
  lsdf = LineSearches.DifferentiableFunction(f,(x,g)->(g[:]=df(x)))

  # allocation and initial evaluation
  tx = copy(x)
  p = similar(x)
  fx = lsdf.fg!(x,p)

  # search direction and slope
  phi  = -p + sum(p)/length(p) # projection to simplex tangent space
  @show dphi = dot(phi, p)

  # maximally move to the border
  #@show alpha0 = minimum(-x[i] / phi[i] for i in 1:length(x) if phi[i] < 0) 
  #x + a phi = 0

  # initial tape
  lsr = LineSearchResults(eltype(x))
  push!(lsr, 0.0, fx, dphi)

  # run bt
  @show alpha, _, _ = backtracking!(lsdf, x, phi, tx, p, lsr, alpha0)
  @show lsr

  @assert tx == x + alpha*phi
  tx, lsr.value[end]
end

proj(x) = GynC.projectsimplex(x)

function testgync(m; n=3, reg=0.9)
  x = fill(1/length(m.xs), length(m.xs))

  f=GynC.mple_obj(m, reg) # maximization objective
  df=GynC.dmple_obj(m, reg)

  pfm(x) = - f(proj(x)) # proj f to minimize
  dfm(x) = - df(x)

  @show pfm(x)

  for i = 1:n
    x = linesearch(pfm, dfm, x, 1e-5)
    #@show pfm(x)
    x = proj(x)
    @assert proj(x) == x
    @show sum(x.==0)
    #@show sum(proj(x).==0)

    #show pfm(x) + f(x)
    #show pfm(x) + f(proj(x))
    #@show sum(x .<= 1e-7)
    @show sort(x)[1:5]
    @show sort(x)[end-4:end]
  end

  @show pfm(x)
  x
end
