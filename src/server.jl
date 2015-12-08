function runsim(out::AbstractMatrix=zeros(1_000,115), person=1)
  sim = mcmc(createsim(MCMCConfig(), ModelConfig())..., 1, verbose=false)

  maxiters = size(out,1)
  blocksize = 10
  
  while last(sim) < maxiters
    a = last(sim) 
    b = min(a+blocksize, maxiters)
    sim = mcmc(sim, b-a, verbose=false)
    #println(typeof(sim))
    out[a:b,:] = sim.value[a:b,:,1]
  end
  out
end

function runsims(persons=1, chains=1, maxiters=10_000)   
  np = length(persons)
  S = SharedArray(Float64, (maxiters, 115, chains, np))
  # possibility 1
  range = [(ip,p,c) for (ip,p) in enumerate(persons), c=1:chains]
  #@parallel for (ip,p,c) in range
  #  runsim(sub(S, (:, :, c, ip)), p)
  #end
  ref = @spawn pmap(r->let ip=r[1], p=r[2], c=r[3]
                   runsim(sub(S, (:,:,c,ip)), p)
                 end, range)
  S, ref
end

