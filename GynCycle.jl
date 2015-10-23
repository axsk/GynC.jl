using MAT

include("utils.jl")

function test_gync()
  parmat = matread("parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  
  tspan = collect(1:0.1:56.0)
  yall  = zeros(length(y0),length(tspan))
  @time gync(yall, y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', yall[[2,7,24,25],:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end
