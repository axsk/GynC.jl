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

function runsim(out::AbstractMatrix=zeros(2_000,115), person=1, c=Channel())
  maxiters = size(out,1)
  blocksize = 100 # interval for updating results
  
  sim = mcmc(createsim(ModelConfig(person))..., 1, verbose=false)

  while last(sim) < maxiters
    a = last(sim) 
    b = min(a+blocksize, maxiters)
    sim = mcmc(sim, b-a, verbose=false)
    #put!(c, sim.value[a:b,:,1]) 
    #save("out/$person.jld", "chain", sim.value)
    out[a:b,:] = sim.value[a:b,:,1]
  end
  out
end

function runsim(iters::Int, person::Int, c::Channel)
  chunksize = 100 # interval for updating results

  sim = mcmc(createsim(ModelConfig(person))..., 1, verbose=false)
  put!(c, sim.value)

  while last(sim) < iters
    a = last(sim)
    b = min(a+blocksize, iters)
    sim = mcmc(sim, b-a, verbose=false)
    put!(c, sim.value[a+1:b,:,1]) 
  end
end

function createjld(filename, iters, chains, persons)
  j = jldopen(filename, "w")
  for p in persons 
    d_create(j.plain, "p$p", Float64, ((iters,115,chains),(-1,115,chains)), "chunk", (100,115,1)) 
  end 
  j
end

function runsims(persons=1, chains=1, iters=1_000)
  np = length(persons)
  channels = [Channel() for i=1:np, j=1:chains]
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  ref = @spawn pmap(range) do r
    let ip=r[1], p=r[2], c=r[3]
      runsim(iters, p, channels[ip, c])
    end
  end
  ref, channels
end




function runsims(persons=1, chains=1, maxiters=1_000)   
  np = length(persons)
  S = SharedArray(Float64, (maxiters, 115, chains, np))
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  ref = @spawn pmap(r->let ip=r[1], p=r[2], c=r[3]
                   runsim(sub(S, (:,:,c,ip)), p)
                 end, range)
  block && wait(ref)
  S, ref
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
