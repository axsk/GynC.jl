const sigma_proposal = 0.1
const chunksize = 1000

typealias Subject Int
id(s::Subject) = string(s)

scaledprop(relprop::Float64, n::Int) = log(1+(relprop^2)) * eye(n)

function startmcmc(subj::Subject, iters::Int, chains::Int, path::String, relprop::Float64=0.1)
  c = ModelConfig(subj)

  # create sampler
  mle_sparms = c.mle_parms[c.sampleparms]
  prop = scaledprop(relprop, length(vcat(mle_sparms, c.mle_y0)))
  samplers = [AMM([:sparms, :y0], prop, adapt=:all)]
 
  # create model
  m = model(c)
  setsamplers!(m, samplers)
  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => c.mle_y0, :sparms => mle_sparms, :data => c.data)
  
  # initial run
  sim = mcmc(m, inp, [ini for i=1:chains], iters, verbose=false)

  path = joinpath(path,"$(id(s)).jld")
  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((iters,115,chains),(-1,115,-1)), "chunk", (chunksize,115,1))
    j["chains"] = sim.value
    j["tune"] = sim.samplers[1].tune
    j["subj"] = subj
  end
end

function continuemcmc(path::String, iters::Int)
  jldopen(path, "r") do j
    last = j["chains"][end, :, :]
    tune = j["tune"] 
    subj = j["subj"]
  end

  c = ModelConfig(subj)
  samplers = [AMM([:sparms, :y0], tune.SigmaF, adapt=:all)]
  m = model(c)
  setsamplers!(m, samplers)
  inp = Dict{Symbol,Any}()
  inis = [Dict{Symbol,Any}(:sparms => last[1,1:88,chain], :y0 => last[1,89:end,chain], :data => c.data) for chain in 1:size(last,3)]

  sim = mcmc(m, inp, inis, iters, verbose=false)

  jldopen(path, "r+") do j
    s = size(j["chains"])
    set_dims!(j["chains"], (s[1]+iters, s[2], s[3]))
    j["chains"][s[1]+1:end, :, :] = sim.values
    j["tune"] = sim.samplers[1].tune
  end
end
