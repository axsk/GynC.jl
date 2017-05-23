using LineSearches

LS_SIMPLEXMINDIST = 1e-6
LS_RELBNDSTEP = 1 # relative stepsize to boundary (0 = disable check)
LS_GRADINT = false # evaluate gradient only in the interior

# the projection to use
proj(x) = GynC.projectsimplex(x)

#= wrong, since sum(phi) != 0
# when on/near border ignore directions outwards
function removeoutward(x, phi)
  phi = copy(phi)
  removes = 0
  for i in eachindex(phi)
    if phi[i] < 0 && x[i] <= LS_SIMPLEXMINDIST
      phi[i] = 0
      removes += 1
    end
  end
  @show removes
  phi
end
=#

#= projection can move to new boundarys?
function removeoutward(x, phi)
  phi=copy(phi)
  n = length(phi)
  a = 1 / sqrt( n-1+(n-1)^2)
  b = a * (1-n)
  N = fill(a, n)

  I = find(i->(phi[i] < 0 && x[i] <= LS_SIMPLEXMINDIST), 1:length(phi))
  for i in I
    n[i] = b
    phi  -= dot(phi, n) * n
    n[i] = a
  end
  @show sum(phi)
  phi
end
=#

function removeoutward(x,phi)
  scale = Inf

  for i = 1:length(phi)
    if (phi[i] < 0 && x[i] > LS_SIMPLEXMINDIST)
      scale = min(scale, -x[i] / phi[i])
    elseif (phi[i] > 0 && x[i] < 1)
      scale = min(scale, (1-x[i]) / phi[i])
    end
  end

  @show scale

  (proj(x + scale*phi) - proj(x)) / scale
end

# stepsize to arrive at border: x + a phi = 0
function alphamaxtoborder(x, phi)
  alpha0=Inf
  for i in 1:length(x)
    if phi[i] < 0 && x[i] > LS_SIMPLEXMINDIST
      alpha0 = min(alpha0, -x[i] / phi[i])
    end
  end
  alpha0
end

function projsimplexint(x)
  y = proj(x)
  i0 = find(x->x==0, y)
  @show length(i0)
  y[i0] .+= SIMPLEXMINDIST
  y / sum(y)
end

projsimplextangent(x) = x - (sum(x) / length(x))
  

" do a minimizing linesearch for given f, df from x in direction of the gradient orthogonally projected onto the simplex "
function simplexlinesearch(f,df,x, alpha0 = 1)
  # start in simplex
  #x = proj(x) 
  
  #@assert all(abs(proj(x) - x) .< 1e-16)

  # restrict f to simplex evaluations
  pf(x) = f(proj(x))
  #pf(x) = f(x)

  function pdf(x) 
    # restrict df to simplex interior evaluations
    x = LS_GRADINT ? projsimplexint(x) : proj(x)
    df(x)
  end
  #pdf(x) = df(x)

  # initialisation
  lsdf = LineSearches.DifferentiableFunction(pf,(x,g)->(g[:]=pdf(x)))

  # allocation and initial evaluation
  p = similar(x)
  fx = lsdf.fg!(x,p)
  @assert all(isfinite(p))
  
  #@show p

  # search direction and slope
  phi = -p # move in opposite direction of gradient
  phi = projsimplextangent(phi) # project onto tangent
  phi = removeoutward(x, phi) # remove outward facing on bnd
  @show norm(phi), maximum(phi)

  # search slope
  #@show phi, p

  @show dphi = dot(phi, p)
  (abs(dphi) <= abs(dot(-p, p))) || warn("dphi larger then df")

  # maximally move to the border: x + a phi = 0
  if LS_RELBNDSTEP > 0
    alpha0 = min(alpha0, alphamaxtoborder(x, phi) * LS_RELBNDSTEP)
  end

  #@show sort(x )[1:5]
  #@show sort(x +alpha0*phi)[1:5]

  #@show maxstep = norm(alpha0 * phi)
  #@show maxcomp = maximum(abs(alpha0 * phi))

  # initialize tape
  lsr = LineSearchResults(eltype(x))
  push!(lsr, 0.0, fx, dphi)

  tx = copy(x)
  @show alpha, _, _ = backtracking!(lsdf, x, phi, tx, p, lsr, alpha0)
  #@show lsr

  #@show alpha
  #@show sum(x+alpha0*phi .<= 1e-8)

  proj(x + alpha * phi)
  #x + (alpha * phi)
end


randomweights(n)  = normalize!(rand(n), 1)

global currentx

function linesearch(m; n=1, reg=0.9, w0 = :uniform)
  
  if w0 == :random
    w0 = randomweights(length(m.xs))
  elseif w0 == :uniform
    w0 = uniformweights(length(m.xs))
  end


  x = Base.normalize(w0, 1)

  #x=Base.normalize(rand(length(m.xs)),1)
  f=GynC.mple_obj(m, reg) # maximization objective
  df=GynC.dmple_obj(m, reg)

  nf(x) = - f(x) # proj f to minimize

  dnf(x) = - df(x)

  @show nf(x)

  for i = 1:n
    nx = simplexlinesearch(nf, dnf, x)
    @show nf(nx)
    @show find(x.<=LS_SIMPLEXMINDIST)

    @assert nf(nx) < nf(x)
    x = nx

    i%10 == 1 && (plot(sort(x)) |> display)

    @assert sum(x.<0) == 0
    @assert sum(x.>1) == 0
    @assert abs(sum(x) - 1) < 1e-5

  end

  x
end

