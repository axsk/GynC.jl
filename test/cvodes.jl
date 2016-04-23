module Sensivity
import Sundials
import Sundials: realtype, N_Vector
import ForwardDiff

type UserData
    rhs!::Function # the main problem rhs
    p::Vector{Float64} # parameters
    Jac::Function # differentiator
end

function extract(userdata_ptr::Ptr{Void})
    d = unsafe_pointer_to_objref(userdata_ptr) :: UserData
    (d.rhs!, d.p, d.Jac)
end

# function used to compute rhs 
function cvodes_rhs(t, y::N_Vector, dy::N_Vector, userdata_ptr::Ptr{Void})
    #@time y   = Sundials.asarray(y)
    #dy  = Sundials.asarray(dy)
    rhs!, p, D! = extract(userdata_ptr)
    rhs!(t, Sundials.asarray(y), p, Sundials.asarray(dy))
    return Int32(0)
end

# given a function `f!(t, y, p, dy)` return the jacobian operator `J: (y0,p0) |-> D_{y,p} f(y0,p0)
function differentiator(f!,ny,np)
    y  = Vector(ny)
    p  = Vector(np)
    dy = Vector(ny)
    J  = Matrix{Float64}(ny, ny+np) # closing over memory
    function f_merged(x)
        f!(0, x[1:ny], x[ny+(1:np)], dy) # TODO fix time dependence
        dy
    end
    # create a mutating jacobian operator for the merged (x and p) problem
    j! = ForwardDiff.jacobian(f_merged, mutates=true)
    
    cy = copy(y) # position of evaluation
    cp = copy(p) 

    ev = Vector{Float64}(ny+np)

    (y, p) -> begin
        ev[1:ny] = y
        ev[ny+(1:np)] = p
        j!(J, ev)
    end
end

NVecVec = Ptr{N_Vector}

function cvodes_sens_rhs(ns::Int32, t::realtype, y::N_Vector, ydot::N_Vector, yS::NVecVec, ySdot::NVecVec, user_data::Ptr{Void}, tmp1::N_Vector, tmp2::N_Vector)
    np = ns
    
    f!, p, D! = extract(user_data)
    y     = Sundials.asarray(y)
    yS    = pointer_to_array(yS, np)
    ySdot = pointer_to_array(ySdot, np)
    
    J = D!(y, p)
    
    ny = size(J,1)
    
    for i in 1:np
        #ySi      = Sundials.asarray(yS[i])
        ySdot[i] = Sundials.nvector(J[:,1:ny] * Sundials.asarray(yS[i]) + J[:,ny+i])
    end
    return Int32(0)
end

### CVODES documentation: https://computation.llnl.gov/casc/sundials/documentation/cvs_guide/

# expect function signature f=f!(t, y0, p, dy)
function cvodes(f::Function, y0::Vector{Float64}, p::Vector{Float64}, ts::Vector{Float64};
    autodiff=true,
    stiff=true,
    solvertype=:dense,
    reltol=1e-6,
    abstol=1e-6,
    ism = Sundials.CV_STAGGERED, # sensitivity solution method
    errconS = 1, 
    reltolS = reltol, 
    abstolS = abstol)
    
  ny   = length(y0)
  np   = length(p)
    
    
  ### 5.5.1 initialization 
  
  # specify linear multistep method / nonlinear solver iteration
  cvode_mem = stiff ? 
    Sundials.CVodeCreate(Sundials.CV_BDF,   Sundials.CV_NEWTON) :
    Sundials.CVodeCreate(Sundials.CV_ADAMS, Sundials.CV_FUNCTIONAL)

  # initialize problem with cvodes_rhs wrapper (calling f) and t0, y0
  Sundials.CVodeInit(cvode_mem, cvodes_rhs, ts[1], y0)
    
  # specify linear solver for newton iteration
  if solvertype == :dense
    Sundials.CVDense(cvode_mem, ny)
  elseif solvertype == :diag
    Sundials.CVDiag(cvode_mem)
  elseif solvertype == :spgmr
    Sundials.CVSpgmr(cvode_mem, 0, 0)
  else
    error("no valid `newton_solver` specified")
  end
  # initialize automatic differentiator
  J! = autodiff ? differentiator(f, ny, np) : ()->()
    
  # Store `f, p, J!` in UserData for evaluation via cvodes_rhs wrapper
  Sundials.CVodeSetUserData(cvode_mem, UserData(f,p,J!))
    
  # specify tolerances
  Sundials.CVodeSStolerances(cvode_mem, reltol, abstol)
    
  # initialize sensivities vector
  yS = [Sundials.nvector(zeros(Float64, ny)) for i in 1:np] |> pointer
  sens_rhs_ptr = autodiff ? 
    sens_rhs_ptr = cfunction(cvodes_sens_rhs, Int32, (Int32, realtype, N_Vector, N_Vector, NVecVec, NVecVec, Ptr{Void}, N_Vector, N_Vector)) :
    Ptr{Void}(0) # use difference quotients
    
  ### 5.2.1 initialize cvode sensivity analysis
  Sundials.CVodeSensInit(cvode_mem, np, ism, sens_rhs_ptr, yS)
    
  ### 5.2.2 Forward sensivity tolerance specification
  
  # scalar relative and absolute tolerances    
  Sundials.CVodeSensSStolerances(cvode_mem, reltolS, ones(np) * abstolS)
  
  # estimate tolerances based on state variable tolerances and scaling factors p^bar
  #Sundials.CVodeSensEEtolerances(cvode_mem)
  
  ### 5.2.5 Optional input for sensivity analysis
  
  # specify problem parameter information for sensivity calculations (p^bar for EEtolerances)
  #Sundials.CVodeSetSensParams(cvode_mem, p, p, Ptr{Int32}(0))
    
  # set difference quotient method
  #Sundials.CVodeSetSensDQMethod(cvode_mem, Sundials.CV_CENTERED, 0.0)

  # specify if sensivity variables are included in the error control mechanism
  Sundials.CVodeSetSensErrCon(cvode_mem, errconS)

  # memory for solution and sensitivities
  solution = zeros(length(ts), ny)
  solution[1,:] = copy(y0)
  sens = zeros(length(ts),ny,np) # No need to copy initial condition, they are already zero

  tout = [0.] # output time reached by the solver
  yout = copy(y0)

  # Loop through all the output times
  for k in 2:length(ts)
    # Extract the solution to x, and the sensitivities to yS
    Sundials.CVode(cvode_mem, ts[k], yout, tout, Sundials.CV_NORMAL)
    Sundials.CVodeGetSens(cvode_mem, tout, yS)

    solution[k,:] = yout
    for i in 1:np
      sens[k,:,i] = Sundials.asarray(unsafe_load(yS,i))
    end
  end

  return (solution, sens)
end

end # module
# output (5.079k) -------------

# code cell -------------------
function f(t,y,p,dy)
    dy[1] = p[1] 
    dy[2] = p[2]^2 * y[1]
end

function test_simple()
  Sensivity.cvodes(f, [0., 0.], [5., 1.], collect(0.:1.:10.), autodiff = true)
end

# output (0.777k) -------------
#=
# code cell -------------------
using GynC
include("../src/gyncycle.jl")

p = copy(GynC.refallparms)
y0 = copy(GynC.refy0)

dy = Array(Float64,33)
D! = differentiator((t, y, p, dy) -> gyncycle_rhs!(y, p, dy), 33, 114);

# code cell -------------------
ind = [3]

yscales = maximum(sol_cvode[:,:], 1) |> vec
tempp = convert(Vector{Any}, p)

function fill(pp, ind)
    allp[ind] = pp    
    allp::Vector{Any}
end

t=[0:1.:31]

function scaledsystem(t,ys,ps,dy)
    y = ys .* yscales
    tempp[ind] = ps .* p[ind]
    gyncycle_rhs!(y, tempp, dy)
    dy[:] = dy[:] ./ yscales
    dy
end

@time sol_cvodes, sens = cvodes(scaledsystem, y0./yscales, ones(length(ind)), t, autodiff=true, stiff=true, solvertype=:dense, reltol = 1e-5, abstol=1e-5, reltolS = 1e-4, abstolS = 1e-5, errconS = 1)

sens[1:26,:,:]

# output (2.759k) -------------

# code cell -------------------
# sundials vs limex
@time sol_cvode = Sundials.cvode((t, y, dy) -> gyncycle_rhs!(y, p, dy), y0, t) #, reltol = 1e-2, abstol=1e-2)
@time sol_limex = GynC.gync(y0, t, p);

# output (0.281k) -------------

# code cell -------------------
# compare solutions
using PyPlot
species = 7
PyPlot.plot(t, sol_cvode[:,species], "<", t, sol_cvodes[:,species], ">", t, sol_limex[species,:] |> vec, "^") 

# output (0.379k) -------------
=# 
