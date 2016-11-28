using GynC
using JLD
using KernelDensity
using Plots

### individual plot functions

paperhd() = paperplot(nsamples=500, niter=100, h=5, zmult=50, smoothmult=50)

function paperplot(;nsamples = 100, niter=50, h=5, zmult=10, smoothmult=10, patient=1, kwargs...)
  m    = gyncmodel(samplepi1(nsamples), zmult=zmult)
  ms   = smoothedmodel(m, smoothmult)
  muni = gyncmodel(samplepi0(nsamples), zmult=0)

  # estimate priors
  w0 = ones(length(m.xs)) / length(m.xs)
  ws = Dict{String, Vector{Vector{Float64}}}()

  println("computing npmle")
  @time ws["NPMLE"] = GynC.em(m, w0, niter)

  println("computing dsmle")
  @time ws["DS-MLE"] = GynC.em(ms, w0, niter);

  println("computing mple")
  @time ws["MPLE"]  = GynC.mple(m, w0, niter, 0.9, h)

  #@time ws["Reference Prior"] = GynC.mple(m, w0, niter, 1, h);

  ### plot results

  # pi 0 plot
  winv = inverseweights(muni.xs)
  pi0plot = plotrow([winv], muni; patient=patient, kwargs...)
  pi0plot[end] = plotpi1patient(patient, nsamples=nsamples, ylims=(0,400))
  ncol = length(pi0plot)


  aplots = vcat(pi0plot, map(x->plotrow(ws[x], m; patient=patient, kwargs...), ["NPMLE", "DS-MLE", "MPLE"])...)
  plot(aplots..., size=(1200, 300*3), layout = (Int(length(aplots)/ncol), ncol))
end

" return the plots for one row "
function plotrow(ws, m;
  patient = 4,
  ylimsdens = :auto,
  ylimstraj = (0,400),
  densspecies = [8, 31, 44, 50, 76]
 )
  
  meas = [datas[patient]]

  ts = 0:1/4:30
  trajspecies = 3
  sols = [GynC.forwardsol(x, ts)[:,GynC.measuredinds[trajspecies]] for x in m.xs];

  wpost = bayesposterior(m, meas, ws[end])

  plots = [begin
	     xs = map(x->x[s], m.xs)
	     plotkdeiters(xs, ws, ylims = ylimsdens)
	     plotkde!(xs, wpost, ylims = ylimsdens)
	   end for s in densspecies]


  plottrajdens(ts, sols, ws[end], ylims=ylimstraj)
  push!(plots, plotdatas!(datas, trajspecies, ylims=ylimstraj))

  plottrajdens(ts, sols, wpost, ylims = ylimstraj)
  push!(plots, plotdatas!(meas, trajspecies, ylims=ylimstraj))
  plots
end


function plotpi1patient(patient;
		       nsamples = 100,
		       ylims = :auto,
		       kwargs...)

  s = JLD.load("../data/0911/allsamples.jld")["samples"]
  xs = subsample([s[patient]], nsamples, 100_000)

  ts = 0:1/4:30
  trajspecies = 3
  sols = [GynC.forwardsol(x, ts)[:,GynC.measuredinds[trajspecies]] for x in xs];

  w = ones(length(xs)) / length(xs)
  plottrajdens(ts, sols, w; ylims=ylims, kwargs...)
  plotdatas!([datas[patient]], trajspecies, ylims = ylims)
end



" plot the kde of iterations of w "
function plotkdeiters(xs, ws; kwargs...)
  colors = colormap("blues", length(ws)+1)[2:end]'
  p = plot(legend=false; kwargs...)
  for (w,c) in zip(ws, colors)
    plotkde!(xs, w; seriescolor = c)
  end
  p
end

const KDEBANDWIDTHMULT = 0.3

function plotkde!(xs, w; kwargs...)
  bw = KernelDensity.default_bandwidth(xs) * KDEBANDWIDTHMULT
  k = kde(xs, weights=w, bandwidth=bw)
  plot!(k.x, k.density; kwargs...)
end


plottrajdens(ts, sols::Vector, w; kwargs...) = plottrajdens(ts, hcat(sols...),w; kwargs...)


" plot the kde of the trajectories "
function plottrajdens(ts, sols::Matrix, weights::Vector = ones(size(sols,1));
		      ylims = :auto,
		      cquant = 0.98,
		      kwargs...)

  bnd = ylims == :auto ? extrema(sols) : ylims

  kdes = [KernelDensity.kde(filter(x->!isnan(x),sols[t,:]), boundary = bnd, weights=weights) for t in 1:size(sols, 1)]

  ys = kdes[1].x
  dens = hcat([k.density for k in kdes]...)

  clims = (0, quantile(vec(dens), cquant))

  contour(ts, ys, dens, clims=clims, fill=true, seriescolor = :heat, legend=false, kwargs...)
end

" plot the given data "
function plotdatas!(datas, species = 3; kwargs...)
  specdatas = map(d->d[:,species], datas)
  scatter!(0:30, specdatas, color=:blue, legend=false, ms=1; kwargs...)
end


### model generation

" generate a gync likelihoodmodel "
function gyncmodel(xs; zmult = 0, sigma=0.1)
  phi(x) = GynC.forwardsol(x)[:,GynC.measuredinds]
  ys = phi.(xs);

  nonaninds = find(x->!any(isnan(x)), ys)

  xs = xs[nonaninds]
  ys = ys[nonaninds]

  err = GynC.MatrixNormalCentered(repmat(sigma*GynC.model_measerrors' * 10, 31)) # 10 hotfix for static scaling in mode.jl

  zs = map(y->y+rand(err), repmat(ys, zmult));

  m = GynC.LikelihoodModel(xs, ys, zs, datas, err);
end

" smooth the data of the given gync model "
function smoothedmodel(m, smoothmult)
  sigmas = [1*KernelDensity.default_bandwidth(filter(x->!isnan(x),[d[i,j] for d in datas])) for i=1:31, j=1:4]
  smoothkernel = GynC.MatrixNormalCentered(sigmas)

  ms = GynC.smoothdata(m, smoothmult, smoothkernel);

  sigmanew = sqrt.(m.measerr.sigmas .^ 2 + smoothkernel.sigmas .^ 2)
  ms.measerr = GynC.MatrixNormalCentered(sigmanew)
  info("adjusted meas error")

  ms
end

" compute the bayes posterior for the given model, data and prior "
function bayesposterior(m, data, wprior)
  L = likelihoodmat(m.ys, data, m.measerr)
  GynC.emiteration(wprior, L)
end


### utility function for handling samples, data and weights
function samplepi0(nsamples)
  yprior = GynC.priory0(1)
  xs = Vector{Float64}[]
  while length(xs) < nsamples
    x = vcat(GynC.refparms.* rand(82) * 5, rand(yprior), 30)
    !any(isnan(GynC.forwardsol(x))) && push!(xs, x)
  end
  xs
end

function samplepi1(n, burnin=100_000)
  s = JLD.load("../data/0911/allsamples.jld")["samples"]
  xs = subsample(s, n, burnin)
end

" given a vector of samplings, pickout `n` samples after the first `burnin` iters "
function subsample(samplings::Vector{Matrix{Float64}}, n::Int, burnin::Int)
  res = Vector{Vector{Float64}}()
  nsamplings = length(samplings)
  for sampling in samplings
    nsamples = size(sampling, 1)
    step = floor(Int,(nsamples - burnin) / n * nsamplings)
    for i = burnin+1:step:nsamples
      s = sampling[i,:]
      push!(res, sampling[i,:])
    end
  end
  res
end

" given some sampling, compute the weigts from the inverse of the kde to obtain a weighted sampling corresponding to the uniform distribution" 
function inverseweights(xs::Vector, stdmult=30)
  w=1./mykde(xs,xs,stdmult)
  normalize!(w, 1)
end

" compute the kde evaluations at 'evals', given the points 'data'.
 for high dimensions adjust stdmult to tweak the covariance "
function mykde(data, evals, stdmult)
  dim = length(data[1])
  stds = [KernelDensity.default_bandwidth(map(x->x[d], data)) for d in 1:dim] * stdmult
  map(evals) do e
    pdf(MvNormal(e, stds), hcat(data...)) |> sum
  end
end

" load all available data "
function alldatas(; minmeas=20)
  datas = vcat([Lausanne(i).data for i=1:40])#, [GynC.Pfizer(i).data for i=1:13])
  datas = filter(d->length(d) - sum(isnan(d)) > minmeas, datas)
end

global datas = alldatas();
