using ClusterManagers

function batch(cs::Vector{Config}, maxiters;
  dir="/nfs/datanumerik/bzfsikor/batch",
  paths = [joinpath(dir, filename(c)) for c in cs],
  maxprocs::Int = 64,
  args...)
  
  nprocs = min(length(cs), maxprocs)
  procs = addprocs(SlurmManager(nprocs), partition="lowPrio")
  try
    eval(Main, :(@everywhere using GynC))
    @everywhere blas_set_num_threads(1)

    isdir(dir) || mkdir(dir)

    res = pmap((c,p) -> GynC.batch(c, maxiters, p; args...), cs, paths)
  finally
    rmprocs(procs)
  end
end


" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."

function batch(c::Config, maxiters, path; batchiters = div(maxiters, 10), overwrite=false)

  local s
  @assert c.thin <= batchiters

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

