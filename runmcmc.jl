using JLD

type ModelConfig
  parms
  y0
  data
  sigma_rho
  sigma_y0
  sigma_parms
  sampleparms
end

function ModelConfig(person=1)
  parms, y0 = loadparms()
  data = loadpfizer()[person]
  sampleparms = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
  ModelConfig(parms, y0, data, 0.1, 1, 20, sampleparms)
end

type MCMCConfig
  iters
  blocksize
  chains
  sampler
  samplernodes
  adapt
  sigma_proposal
end

MCMCConfig() = MCMCConfig(1_000, 100, 3, :amm, [:sparms, :y0], true, 0.1)
  


function runmcmc(c::Config)
  jldname = "$sampler $adapt ugm-flat $(length(persons))pX$(chains)c $(SIGMA_RHO)r $(SIGMA_Y0)sy $(SIGMA_PARMS)sp $(SIGMA_PROPOSAL)p.jld"

  proposalvariance = log(1+(c.sigma_proposal^2)) * eye(length(c.sampleparms)+length(c.y0))

  sampler = 
    c.sampler == :amm ? [AMM(c.samplingnodes, proposalvariance, adapt=c.adapt)] :
    c.samper  == :amwg ? [AMWG(c.samplingnodes, diag(proposalvariance))] :
    error("invalid sampler specified")

  m, inp, ini = gyncmodel(c.data, c.parms, c.y0)
  setsamplers!(m, samplers[sampler])

  mc = mcmc(m, inp, [ini[1] for i=1:c.chains], c.iterblock, verbose = false, chains=c.chains)

  println("writing to $jldname")

  while size(mc,1) < iters
    @time for i in 1:length(persons)
    mc = mcmc(mc, c.blocksize, verbose=false)
    save(jldname, "mc", mc.value) 
  end
end
