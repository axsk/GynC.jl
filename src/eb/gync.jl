# glue code to generate a likelihoodmodel for eb from gync samples

function gyncmodel(n::Int;kwargs...)
  xs = samplepi1(n)
  datas = alldatas()
  gyncmodel(xs, datas; kwargs...)
end

" generate a gync likelihoodmodel "
function gyncmodel(xs, datas; zmult = 0, sigma=0.1)
  phi(x) = GynC.forwardsol(x)[:,GynC.measuredinds]
  ys = phi.(xs);

  nonaninds = find(x->!any(isnan(x)), ys)
  length(nonaninds) > 0 && warn("removed some samples since they lead to NaN results")

  xs = xs[nonaninds]
  ys = ys[nonaninds]

  err = GynC.MatrixNormalCentered(repmat(sigma*GynC.model_measerrors' * 10, 31)) # TODO: 10 hotfix for static scaling in model.jl

  zs = map(y->y+rand(err), repmat(ys, zmult));

  m = GynC.LikelihoodModel(xs, ys, zs, datas, err);
end

## load saved samples

global BURNIN = 100_000

function loadallsamples()
  allsamplepath = "/datanumerik/bzfsikor/gync/0911/allsamples.jld"
  JLD.load(allsamplepath, "samples") :: Vector{Matrix{Float64}}
end


" given a vector of samplings, pickout `n` samples after the first `burnin` iters "
function samplepi1(n, burnin=BURNIN)
  xs = subsample(loadallsamples(), n, burnin)
end

function samplepi1rnd(n, burnin=BURNIN)
  s = loadallsamples()
  nsamplings = length(s)
  res = Vector{Vector{Float64}}()

  for i = 1:n
    sampling = s[i%nsamplings+1]
    j = rand((BURNIN+1):size(sampling,1))
    push!(res, sampling[j, :])
  end
  res
end

function samplepi0(nsamples, trajts=0:30)
  yprior = GynC.priory0(1)
  xs = Vector{Float64}[]
  while length(xs) < nsamples
    x = vcat(GynC.refparms.* rand(82) * 5, rand(yprior), 30)
    !any(isnan(GynC.forwardsol(x, trajts))) && push!(xs, x)
  end
  xs
end

" produce n evenly spaced subsamples (given a vector of matrices where each row contains a sample) "
function subsample(samplings::Vector{Matrix{Float64}}, n::Int, burnin::Int)
  res = Vector{Vector{Float64}}()
  nsamplings = length(samplings)
  for sampling in samplings
    nsamples = size(sampling, 1)
    step = floor(Int,(nsamples - burnin) / n * nsamplings)
    for i = burnin+1:step:nsamples
      push!(res, sampling[i,:])
    end
  end
  res[1:n]
end

