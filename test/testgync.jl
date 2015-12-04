""" load parameters and initial data and run a cycle with gync """
function test_gync()
  parms, y0 = loadmles() 
  tspan = collect(1:0.1:56.0)
  @time y = gync(y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', y[MEASURED,:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end
