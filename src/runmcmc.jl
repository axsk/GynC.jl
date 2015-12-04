type MCMCConfig
  iters                 # number of total iterations
  blocksize             # iterations between saves
  chains
  sampler               # :amm or :amwg
  adapt
  sigma_proposal        # std relative to current value
end

MCMCConfig() = MCMCConfig(1_000, 100, 3, :amm, true, 0.1)

function runmcmc(gc::GynCConfig, mc::MCMCConfig, jldname="test.jld")
  # TODO: proof
  proposalvariance = log(1+(c.sigma_proposal^2)) * eye(length(c.sampleparms)+length(c.y0))
  sampler = 
    mc.sampler == :amm ?  [AMM([:sparms, :y0],  proposalvariance, adapt=c.adapt)] :
    mc.samper  == :amwg ? [AMWG([:sparms, :y0], diag(proposalvariance))] :
    error("invalid sampler specified")

  m = model(gc)
  setsamplers!(m, sampler)

  mle_sparms = gc.mle_parms[gc.sampleparms]
  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => gc.mle_y0, :sparms => mle_sparms, :data => gc.data)

  mc = mcmc(m, inp, [ini for i=1:mc.chains], mc.blocksize, verbose=false, chains=mc.chains)

  !isempty(jldname) && println("writing to $jldname")

  while size(mc,1) < iters
    !isempty(jldname) && save(jldname, "chain", mc.value, ") 
    @time mc = mcmc(mc, mc.blocksize, verbose=false)
  end
  mc
end
