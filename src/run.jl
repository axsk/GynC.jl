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

function runsim(out::AbstractMatrix=zeros(1_000,115), person=1)
  maxiters = size(out,1)
  blocksize = 100
  
  sim = mcmc(createsim(ModelConfig(person))..., 1, verbose=false)

  while last(sim) < maxiters
    a = last(sim) 
    b = min(a+blocksize, maxiters)
    sim = mcmc(sim, b-a, verbose=false)
    save("out/$person.jld", "chain", sim.value)
    out[a:b,:] = sim.value[a:b,:,1]
  end
  out
end

function runsims(persons=1, chains=1, maxiters=1_000, block=false)   
  np = length(persons)
  S = SharedArray(Float64, (maxiters, 115, chains, np))
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  ref = @spawn pmap(r->let ip=r[1], p=r[2], c=r[3]
                   runsim(sub(S, (:,:,c,ip)), p)
                 end, range)
  block && wait(ref)
  S, ref
end

function benchmark(chains=nworkers(), maxiters=1000)
  tic()
  S,r=runsims(1,chains,maxiters)
  wait(r)
  println(chains, " chains: ", chains*maxiters/toc(), " samples/s")
end

function script()
  S, ref = runsims(1:5,3,1_000, true)
  save("out/all.jld", "chains", S)
end
