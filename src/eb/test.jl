n=2000
srand(1)
L=rand(n,n)
x0 = rand(n)
x0 = x0 / sum(x0)

function hz(wx, Lzx::Matrix=L, wz=wx)
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

function logl(wx, Ldx::Matrix=L)
  sum(log(Ldx*wx))
end

r = 0.95
function h(x)
  let x=x/sum(x) 
  -(hz(x/sum(x)) * r + (1-r) * logl(x/sum(x))) 
  end
end

proj(x) = let o = normalize!(ones(x))
  x - dot(x, o) * o
end

using ReverseDiff

global evals = 0

dh! = ReverseDiff.compile_gradient(h,x0)
function dhp!(g,x) 
  dh!(g,x)
  #g[:] = proj(g)
end

function myfunc(x, grad) 
  global evals+=1
  dhp!(grad,x)
  h(x)
end

function myconst(x, grad)
  grad[:] = ones(grad)
  sum(x)-1
end

function myconst2(x,grad)
  grad[:] = -ones(grad)
  1-sum(x)
end


using NLopt

function nlopt(x0)

  opt = Opt(:LD_MMA,n)
  #inequality_constraint!(opt, myconst, 1e-3)
  #inequality_constraint!(opt, myconst2, 1e-3)

  #=opt = Opt(:LD_AUGLAG,n)
  local_optimizer!(opt, Opt(:LD_MMA,n))
  equality_constraint!(opt, myconst)
  =#
  global evals = 0

  lower_bounds!(opt, 0.)
  upper_bounds!(opt, 1.)
  min_objective!(opt, myfunc)
  #ftol_rel!(opt, 1e-13)
  maxeval!(opt, 100)
  xtol_rel!(opt, 0.001)
  xtol_abs!(opt, 1e-9)

  @time minf, minx, ret = optimize(opt, x0)
  @show minf
  @show ret
  @show evals

  #minx = minx/sum(minx)
  @assert all(minx .>= 0) && all(minx .<= 1)
  minx / sum(minx)
end


#=using JuMP


m=Model(solver=NLoptSolver(algorithm=:LD_MMA))
@variable(m, 0<=x[1:n]<=1)

myh(x...) = h([x...])
mydh(g, x...) = dh!(g, [x...])
JuMP.register(:myobj, n, myh, mydh)
@constraint(m, sum(x) == 1)
@NLobjective(m, Min, myobj(x[1],x[2],x[3]))

solve(m)
=#

#=
using Optim

Optim.optimize(OnceDifferentiable(h, (x,g) -> begin
                                  dh!(g,x)
                                  g[:] = proj(g)
                                  g
                                  end), x0, zeros(x0), ones(x0), Fminbox(), optimizer = GradientDescent)
=#
