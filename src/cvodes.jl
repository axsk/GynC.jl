module Sensivity

import ForwardDiff
import Sundials
import Sundials: realtype, N_Vector
NVecVec = Ptr{N_Vector}

# userdata object from sundials, storing our problem
type UserData
    f!::Function # the main problem rhs
    p::Vector{Float64} # parameters
    jac::Function # differentiator
end

extract(userdata_ptr::Ptr{Void}) = unsafe_pointer_to_objref(userdata_ptr)

# return the jacobian function for f
function differentiator(f!,ny,np)
    dy = Vector(ny)
    Jy = Matrix{Float64}(ny,ny)
    Jp = Matrix{Float64}(ny,np)
    # add cache?
    cachey = ForwardDiff.ForwardDiffCache()
    cachep = ForwardDiff.ForwardDiffCache()

    # Return function which for given `t, y, p` returns the jacobians wrt y and p: `Jy, Jp`.
    function jac(t, y, p)
        Jy = ForwardDiff.jacobian(y->(f!(t,y,p,dy); dy), y, cache=cachey)
        Jp = ForwardDiff.jacobian(p->(f!(t,y,p,dy); dy), p, cache=cachep)
        Jy, Jp
    end
end

function cvodes_sens_rhs(ns::Int32, t::realtype, y::N_Vector, ydot::N_Vector, yS::NVecVec, ySdot::NVecVec, user_data::Ptr{Void}, tmp1::N_Vector, tmp2::N_Vector)
    np = ns
    
    u = extract(user_data)
    Jy, Jp = u.jac(t, Sundials.asarray(y), u.p)
    
    yS    = pointer_to_array(yS, np)
    ySdot = pointer_to_array(ySdot, np)

    for i in 1:np
        ySdot[i] = Sundials.nvector(Jy * Sundials.asarray(yS[i]) + Jp[:,i])
    end
    return Int32(0)
end

# function used to compute rhs 
function cvodes_rhs(t, y::N_Vector, dy::N_Vector, userdata_ptr::Ptr{Void})
    u = extract(userdata_ptr)
    f! = u.f!
    p  = u.p
    f!(t, Sundials.asarray(y), p, Sundials.asarray(dy))
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
  elseif solvertype == :spbcg
    Sundials.CVSpbcg(cvode_mem, 0, 0)
  elseif solvertype == :sptfqmr
    Sundials.CVSptfqmr(cvode_mem, 0, 0)
  else
    error("no valid `newton_solver` specified")
  end

  # initialize automatic differentiator
  jac = autodiff ? differentiator(f, ny, np) : ()->()
    
  # Store `f, p, J!` in UserData for evaluation via cvodes_rhs wrapper
  Sundials.CVodeSetUserData(cvode_mem, UserData(f,p,jac))
    
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
