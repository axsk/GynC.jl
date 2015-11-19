include("gyncycle.jl")
tspan = Array{Float64}(collect(1:31))
parms, y0 = loadparms()
@assert length(unique([y = gync(y0, tspan, parms) for i=1:100])) == 1
