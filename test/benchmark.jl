using BenchmarkTools
using GynC: gyncycle_rhs!, gync, *

@show @benchmark GynC.gyncycle_rhs!(GynC.refy0, GynC.refallparms, similar(GynC.refy0))
@show @benchmark gync(GynC.refy0, GynC.refallparms, 0:1:30)
