using ClusterManagers

function batch(cs::Vector{Config}, maxiters; dir="", maxprocs::Int = 8)
  
  # initialize slurm processes
  nprocs = min(length(cs), maxprocs)
  try
    procs = addprocs(nprocs)
    @everywhere require("GynC")
    @everywhere blas_set_num_threads(1)

    paths = [joinpath(dir,configpath(c)) for c in cs]

    res = pmap(c->GynC.batch(c, 10), cs, paths)
  finally
    rmprocs(procs)
  end
end

filename(c::Config) = "p$(c.patient.id)s$(c.sigma_rho)r$(c.relprop)t$(c.thin).jld"



" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."

function batch(c::Config, maxiters, path; batchiters = div(maxiters, 10), overwrite=false)

  local s

  if !isfile(path) || overwrite
    info("writing to $path")
    isa(c, Config) || Base.error("Need to give a config")
    s = sample(c, min(batchiters,maxiters))
    save(path, s)
  else
    info("resuming $path")
    s = load(path)
  end

  thin = c.thin

  while (iters = min(batchiters, maxiters-(thin*size(s.samples, 1)))) >= thin
    s = sample!(s, iters)
    save(path, s)
  end
  s
end

