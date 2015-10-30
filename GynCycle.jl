using MAT, DataFrames

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

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function loadpfizer(path = "data/pfizer_normal.txt")
  data=readtable("data/pfizer_normal.txt", separator='\t')
  results = Vector()
  for patientdata in groupby(data, 6)
    p=fill(NaN, 4, 31)
    for meas in eachrow(patientdata)
      # map days to 1-31
      day = (meas[1]+30)%31+1 
      for i=1:4
        val = meas[i+1]
        p[i,day] = isa(val, Number) ? val : NaN
      end
    end
    append!(results, Any[p])
  end
  results
end

function loadparms()
  parmat = matread("parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  parms, y0
end

""" load parameters and initial data and run a cycle with gync """
function test_gync()
  parms, y0 = loadparms() 
  tspan = collect(1:0.1:56.0)
  @time y = gync(y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', y[[2,7,24,25],:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end

""" likelihood (up to proport.) for the latent parameters given the patientdata """
function loglikelihood(parms, data, y0)
  sigma = 1
  tspan = Array{Float64}(collect(1:31)) 
  y = gync(y0, tspan, parms)
  
  distsq = 0
  simdata = y[[LH,FSH,E2,P4],:]
  
  sre = minimum([squaredrelativeerror(data, translatecol(simdata, transl)) for transl in 0:30])
  -1/(2*sigma^2) * sre
end

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1, data2)
  diff = data1 - data2
  rdiff = diff ./ data1
  sre   = sum(rdiff[!isnan(rdiff)] .^ 2)
end


function translatecol(data,transl)
  hcat(data[:, 1+transl:31], data[:, 1:transl])
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
