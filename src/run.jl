const sigma_proposal = 0.1
const chunksize = 10

typealias Subject Int
id(s::Subject) = string(s)

""" run a single chain and push the updates to the given channel """
function runchain(iters::Int, subj::Subject, update::Union{AbstractChannel, RemoteRef})
  c = ModelConfig(subj)

  # create sampler
  mle_sparms = c.mle_parms[c.sampleparms]
  d = eye(length(vcat(mle_sparms, c.mle_y0)))
  proposalvariance = log(1+(sigma_proposal^2)) * d
  sampler = [AMM([:sparms, :y0],  proposalvariance, adapt=:all)]
 
  # create model
  m = model(c)
  setsamplers!(m, sampler)
  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => c.mle_y0, :sparms => mle_sparms, :data => c.data)
  
  # initial run
  sim = mcmc(m, inp, [ini], min(chunksize, iters), verbose=false)
  put!(update, sim.value[:,:,1])


  # subsequent runs
  while last(sim) < iters
    a = last(sim)
    b = min(a+chunksize, iters)
    sim = mcmc(sim, b-a, verbose=false)
    put!(update, sim.value[a+1:b,:,1]) 
  end
end

""" a job contains multiple running chains for one subject """
function runchains(s::Subject, iters::Int, chains::Int, path="out")
  # initialize file
  path = "$path/$(id(s)).jld"
  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((iters,115,chains),(-1,115,-1)), "chunk", (chunksize,115,1))
  end

  # channels for the updates
  channels = [RemoteRef() for c in 1:chains]
  counter  = zeros(Int, chains)

  # spawn each chain in a thread
  refs = map(c->@spawn(runchain(iters, s, c)), channels)
  
  # create task to safe updates
  for (i,c) in enumerate(channels)
    @schedule while true
      wait(c)
      samples = take!(c)
      jldopen(path, "r+") do j
        j["chains"][counter[i]+1:counter[i]+size(samples, 1), :, i] = samples
      end
      counter[i] += size(samples, 1)
      # race condition?
      isready(refs[i]) && !isready(c) && break
    end
  end
  refs, channels, () -> sum(counter) / (iters*chains)
end

function script(persons=1:3, iters=500_000, chains=3)
  refs = vcat([runchains(s, iters, chains) for s in persons]...)
end

function benchmark(iters=300, chains=nworkers())
  runchains(1, iters, chains, "/tmp/gync")
end
