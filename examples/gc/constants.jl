# indices for measured variables: LH, FSH, E2, P4
const measuredinds = [2,7,24,25]
const hillinds     = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampledinds  = deleteat!(collect(1:103), hillinds)

const refy0       = include("data/refy0.jl")
const refallparms = include("data/refparms.jl")
const refparms    = refallparms[sampledinds]
const refinit     = vcat(refparms, refy0)


const speciesnames   = include("data/speciesnames.jl")
const parameternames = include("data/parameternames.jl")[sampledinds]
const samplednames   = [parameternames; speciesnames]
