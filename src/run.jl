const sigma_proposal = 0.1
const chunksize = 1000

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

""" run mutliple chains asynchronously """
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

  progress() = sum(counter) / (iters*chains)
  refs, channels, progress
end

function run(persons=1, iters=100, chains=1)
  fns=[runchains(s, iters, chains)[3] for s in persons]
  progress() = mean(map(x->x(), fns))
  next = 0
  start = now()
  while true
    if progress() >= next
      next = min(next+0.02, 1)
      println(now(), ", ", progress()*100, "%, ", round(progress()*iters*chains*length(persons)/(Int(now()-start)/1000),1), " s/sec")
    end
    progress() == 1 && break
    sleep(1/20)
  end
end
