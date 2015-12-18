function createsim(c::ModelConfig, sigma_proposal = 0.1)
  mle_sparms = c.mle_parms[c.sampleparms]
  d = eye(length(vcat(mle_sparms, c.mle_y0)))

  proposalvariance = log(1+(sigma_proposal^2)) * d
  sampler = [AMM([:sparms, :y0],  proposalvariance, adapt=:all)]

  m = model(c)
  setsamplers!(m, sampler)

  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => c.mle_y0, :sparms => mle_sparms, :data => c.data)

  m, inp, [ini]
end

type SampleChannel
  channel
  person::Int
  chain::Int
  counter::Int
end

SampleChannel(person::Int, chain::Int) = SampleChannel(RemoteRef(()->Channel()), person, chain, 0)

function runsim(iters::Int, person::Int, sc::SampleChannel)
  chunksize = 1000
  sim = mcmc(createsim(ModelConfig(person))..., chunksize, verbose=false)
  put!(sc.channel, sim.value[:,:,1])
  while last(sim) < iters
    a = last(sim)
    b = min(a+chunksize, iters)
    sim = mcmc(sim, b-a, verbose=false)
    put!(sc.channel, sim.value[a+1:b,:,1]) 
  end
end

typealias Subject Int
id(s::Subject) = String(s)

typealias Chain SampleChannel

""" a job contains multiple running chains for one subject """
function job(s::Subject, iters::Int, chains::Int)
  # initialize file
  filename = "$path/$(id(s)).jld"
  jldopen(filename, "w") do j
    d_create(j.plain, "chains", Float64, ((iters,115,chains),(-1,115,-1)), "chunk", (100,115,1))
  end

  # channels for the updates
  channels = [RemoteRef(()->Channel()) for c in 1:chains]
  counter  = zeros(chains)

  # spawn each chain in a thread
  refs = map(c->@spawn runsim(iters, p, c), channels)
  
  # create task to safe updates
  for (i,c) in channels
    @schedule while true
      wait(c)
      samples = take!(c)
      jldopen(filename, "r+") do j
        j["chains"][counter+1 : counter+size(samples, 1), :, i] = samples
      end
      counter[i] += size(samples, 1)
      # race condition?
      isready(refs[i]) && !isready(c) && break
    end
  end

   
end


function savechannels(channels)
  for c in channels
    !isready(c.channel) && continue
    samples = take!(sc.channel)
    a = sc.counter + 1
    b = sc.counter + size(samples, 1)
    sc.counter = b
    jldopen("out/all.jld", "r+") do j
      j["p$(sc.person)"][a:b,:,sc.chain] = samples
    end
    println("saved ", size(samples,1), " samples")
  end
end


function jldinit(iters, chains, persons)
  jldopen("out/all.jld", "w") do j
    for p in persons 
      d_create(j.plain, "p$p", Float64, ((iters,115,chains),(-1,115,chains)), "chunk", (100,115,1)) 
    end
  end 
end

function runsims(persons=1:5, chains=3, iters=500_000)
  np = length(persons)
  channels = [SampleChannel(p, c) for p in persons, c=1:chains]
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  ref = @spawn pmap(range) do r
    let ip=r[1], p=r[2], c=r[3]
      runsim(iters, p, channels[ip, c])
    end
  end
  jldinit(iters, chains, persons)
  while !isready(ref)
    tic()
    savechannels(channels)
    sampled = sum([sc.counter for sc in channels])
    println(sampled, " samples (", round(100*sampled/(length(persons)*chains*iters),1), "%)")
    sleep(10)
    toc()
  end
  savechannels(channels)
  ref
end

function benchmark(chains=nworkers(), maxiters=200)
  S,r=runsims(1,chains,5)
  tic()
  S,r=runsims(1,chains,maxiters)
  wait(r)
  println(chains, " chains: ", chains*maxiters/toc(), " samples/s")
end

function script(persons=1:5, chains=3, iters=500_000)
  S, ref = runsims(persons, chains, iters)
  saveloop(S, ref)
end

function saveloop(S, ref; filename="out/all.jld", sleeptime = 60)
  save(filename, "chains", Array(S))
  idx = zeros(Int, size(S)[3:4])

  while true
    tic()
    for c=1:size(S,3), p=1:size(S,4)
      idx[c,p] == size(S,1) && continue 
      # get index of last row with no zeros
      nidx = findfirst(i->any(S[i,:,c,p] .==0), 1:size(S,1)) - 1
      nidx == -1 && (nidx = size(S,1))
      nidx == 0  && continue
      nidx == idx[c,p] && continue 
         
      println("writing ", nidx-idx[c,p], " samples to file")
      jldopen(filename, "r+") do file
        file["chains"][idx[c,p]+1:nidx, :, c, p] = S[idx[c,p]+1:nidx, :, c, p]
      end

      idx[c,p] = nidx
    end

    println("wrote ", sum(idx), " samples, (", round(sum(idx) / (size(S,1) * size(S,3) * size(S,4)) * 100, 1), "%)")
  
    all(idx .== size(S,1)) && break
    sleep(sleeptime)
    toc()
  end
end
