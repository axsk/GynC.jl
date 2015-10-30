using MAT, DataFrames

include("utils.jl")

# indices for measured variables: LH, FSH, E2, P4
const MEASURED = [2,7,24,25]
# indices for the parameters to be sampled
const SAMPLEPARMS = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]

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
    melt(DataFrame(vcat(tspan', y[MEASURED],:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end

""" likelihood (up to proport.) for the parameters given the patientdata """
function loglikelihood(data::Matrix, parms::Vector, y0::Vector, sigma_rho::Real)
  tspan = Array{Float64}(collect(1:31)) 
  y = gync(y0, tspan, parms)
  
  distsq = 0
  simdata = y[MEASURED,:]
  
  sre = minimum([squaredrelativeerror(data, translatecol(simdata, transl)) for transl in 0:30])
  -1/(2*sigma_rho^2) * sre
end

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  # TODO: divide by data1 or data2?
  rdiff = diff ./ data1
  sre   = sumabs2(rdiff[!isnan(rdiff)])
end

""" cyclic translation of the columns of `data` by `transl` to the left """
function translatecol(data::Matrix, transl::Integer)
  hcat(data[:, 1+transl:31], data[:, 1:transl])
end

""" Return the Bayesian Model with priors y0 ~ Norm(y0), parms' ~ Norm(parms'). Here parms' denotes the sampled parameters, while `parms` are all parameters. """
function gyncmodel(data::Matrix, parms::Vector, y0::Vector)
  tparms = copy(parms) # copy for mutating via mergeparms!
  sigma_y0 = diagm(y0/10)
  sigma_parms = diagm(parms[SAMPLEPARMS]/10)
  sigma_rho = length(SAMPLEPARMS) / 10

  m = Model(
    y0 = Stochastic(length(y0),
      () -> MVNormal(y0, sigma_y0)),
    sparms = Stochastic(length(SAMPLEPARMS),
      () -> MVNormal(allparms[SAMPLEPARMS], sigma_parms)),
    parms = Logical(length(parms),
      (sparms) -> mergeparms!(sparms, tparms), false),
    data = Stochastic(size(data, 1),
      (y0, parms) -> LogDensityDisribution(data->loglikelihood(data, aparms, y0, sigma_rho))))

  inputs = Dict{Symbol,Any}()
  inits  = Dict{Symbol,Any}(:y0 -> y0, :parms -> parms, :data -> data)
  m, inputs, [inits]
end

function mergeparms!(sampled::Vector, all::Vector)
  all[SAMPLEPARMS] = sampled
  all
end

function test_mcmc()
  parms, y0 = loadparms()
  data = loadpfizer[1]
  mcmc(gyncmodel(data, parms, y0)..., 1000)
end
