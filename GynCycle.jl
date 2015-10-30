using MAT

include("utils.jl")

function matparms()
  parmat = matread("parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  parms, y0
end


function test_gync()
  parms, y0 = matparms() 
  tspan = collect(1:0.1:56.0)
  y     = zeros(length(y0),length(tspan))
  @time gync!(y, y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', yall[[2,7,24,25],:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end

function likelihood(x::Array,y::Array)
    
    176.57.128.3:2520
end
