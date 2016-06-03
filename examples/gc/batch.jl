" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."
function batch(;batchiters=100_000, maxiters=10_000_000, config::Union{Config, Void}=nothing, overwrite=false, path=config == nothing ? "" : configpath(config))

  local s

  if !isfile(path) || overwrite
    info("writing to $path")
    isa(config, Config) || Base.error("Need to give a config")
    s = sample(config, batchiters)
    save(path, s)
  else
    info("resuming $path")
    s = load(path)
  end

  thin = s.config.thin

  while (iters = min(batchiters, maxiters-(thin*size(s.samples, 1)))) >= thin
    s = sample!(s, iters)
    save(path, s)
  end
  s
end

configpath(c::Config) = "p$(c.patient.id)s$(c.sigma_rho)r$(c.relprop)t$(c.thin).jld"
