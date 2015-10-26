using MAT

include("utils.jl")

type PatientData
  time::Vector
  data::Array # (4,length(time))
end

# indices for latent variables
const LH  = 2
const FSH = 7
const E2  = 24
const P4  = 25

function loadpfizer(path = "data/pfizer_normal.txt")
  data = readtable(path, separator = '\t')
  by(data, 6) do patientdata
    PatientData(patientdata[1], patientdata[2:5])
  end
end

function test_gync()
  parmat = matread("parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  
  tspan = collect(1:0.1:56.0)
  @time y = gync(y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', y[[2,7,24,25],:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end

""" likelihood (up to proport.) for the latent parameters given the patientdata """
function loglikelihood(lparms, data::PatientData, y0)
  y = zeros(length(y0), length(data.time))
  
  gync(y, y0, data.time, parms)
  
  distsq = 0
  yobserved = y[[LH,FSH,E2,P4],:]

  diff  = y-data.data 
  rdiff = diff ./ data.data 

  sre   = sum(rdiff .^ 2)

  -1/(2*sigma^2) * sre
end


function patientmodel(data::PatientData, allparms::Vector, y0)
  allparms = copy(allparms)
  Model(
    y0 = Stochastic(MVNormal(y0, sigma)),
    likelihood = Stochastic(
      y0 -> LogDensityDisribution(
        parms->loglikelihood(mergepams!(parms, allparms), data, y0))))
end

function mergeparms!(latent::Vector, fixed::Vector)
  fixed[[LH, FSH, E2, P4]] = latent
  fixed 
end
