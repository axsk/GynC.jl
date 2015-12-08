type MCMCConfig
  iters                 # number of total iterations
  blocksize             # iterations between saves
  chains
  sampler               # :amm or :amwg
  adapt
  sigma_proposal        # std relative to current value
  jldname
end

MCMCConfig() = MCMCConfig(200, 100, 3, :amm, :all, 0.1, "")

function createsim(mc::MCMCConfig, gc::ModelConfig)
  # TODO: proof
  mle_sparms = gc.mle_parms[gc.sampleparms]

  proposalvariance = log(1+(mc.sigma_proposal^2)) * eye(length(mle_sparms)+length(gc.mle_y0))
  sampler = 
    (mc.sampler == :amm) ?  [AMM([:sparms, :y0],  proposalvariance, adapt=mc.adapt)] :
    (mc.samper  == :amwg) ? [AMWG([:sparms, :y0], diag(proposalvariance))] :
    error("invalid sampler specified")

  m = model(gc)
  setsamplers!(m, sampler)

  inp = Dict{Symbol,Any}()
  ini = Dict{Symbol,Any}(:y0 => gc.mle_y0, :sparms => mle_sparms, :data => gc.data)

  chain = (m, inp, [ini])
end

