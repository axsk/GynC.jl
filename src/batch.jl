using ClusterManagers

function batch(cs::Vector{Config}, iters::Vector{Int};
  dir = BATCHDIR,
  paths = [joinpath(dir, filename(c)) for c in cs],
  maxprocs::Int = 64,
  args...)
  
  isdir(dir) || mkdir(dir)
  cd(dir) do
    nprocs = min(length(cs), maxprocs)
    procs = addprocs(SlurmManager(nprocs), partition="lowPrio")
    try
      eval(Main, :(@everywhere using GynC))
      @everywhere blas_set_num_threads(1)
      res = pmap((c,p) -> GynC.batch(c, iters, p; args...), cs, paths)
    finally
      rmprocs(procs)
    end
  end
end


" Sample from `c::Config` and save result for all reached iterations in `iters` to `path` "
function batch(c::Config, iters::Vector{Int}, path::AbstractString; overwrite=false)
  s = Sampling(c)
  !overwrite && isfile(path) && (s = load(path))
  thin = c.thin
  for i in iters
    n = size(s.samples, 1) * thin
    i = i - n
    i < thin && continue
    s = sample!(s, i)
    save(path, s)
  end
  s
end
