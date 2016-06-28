using BenchmarkTools
using GynC: gyncycle_rhs!, gync, *

y = GynC.refy0
p = GynC.refallparms
dy = similar(y)

@show @benchmark GynC.gyncycle_rhs!(y, p, dy)

@show @benchmark gync(y, p, 0:0.1:30)
@show @benchmark gync(y, p, 0:1:30)
@show @benchmark gync(y, p, 0:1:90)
