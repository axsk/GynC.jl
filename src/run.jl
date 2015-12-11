function createsim(gc::ModelConfig, simga_proposal = 0.1)
  mle_sparms = gc.mle_parms[gc.sampleparms]
  d = eye(length(vcat(mle_sparms, gc.mle_y0)))

  proposalvariance = log(1+(sigma_proposal^2)) * d
  sampler = [AMM([:sparms, :y0],  proposalvariance, adapt=true)]

  m = model(gc)
  setsamplers!(m, sampler)

  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => gc.mle_y0, :sparms => mle_sparms, :data => gc.data)

  m, inp, [ini]
end

function runsim(out::AbstractMatrix=zeros(1_000,115), person=1)
  sim = mcmc(createsim(MCMCConfig(), ModelConfig())..., 1, verbose=false)

  maxiters = size(out,1)
  blocksize = 10
  
  while last(sim) < maxiters
    a = last(sim) 
    b = min(a+blocksize, maxiters)
    sim = mcmc(sim, b-a, verbose=false)
    out[a:b,:] = sim.value[a:b,:,1]
  end
  out
end

function runsims(persons=1, chains=1, maxiters=10_000, block=false)   
  np = length(persons)
  S = SharedArray(Float64, (maxiters, 115, chains, np))
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  ref = @spawn pmap(r->let pind=r[1], p=r[2], c=r[3]
                   runsim(sub(S, (:,:,c,pind)), p)
                 end, range)
  block && wait(ref)
  S, ref
end

function benchmark(chains=nworkers(), maxiters=100)
  tic()
  S,r=runsims(1,chains,maxiters)
  wait(r)
  println(chains, " chains: ", chains*maxiters/toc(), " samples/s")
end

function script()
  S, ref = runsims(8,4,500, true)
  save("/datanumerik/bzfsikor/chains.jld", "chains", S)
end
