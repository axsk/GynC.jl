using BenchmarkTools
using GynC: gyncycle_rhs!, gync, *

@show @benchmark GynC.gyncycle_rhs!(GynC.refy0, GynC.refallparms, similar(y))
@show @benchmark gync(GynC.refy0, GynC.refallparms, 0:1:30)
