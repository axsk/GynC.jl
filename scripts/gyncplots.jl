using GynC
using Memoize
using JLD
using KernelDensity
using Plots

pyplot(grid=false)

#const densspecies = [8, 31, 44, 50, 76]
const densspecies = [31]
const denskdebw = 0.3
const denscolor = :orangered

const patient = 4
const sigma = 0.2

const trajspecies = 3
const trajts = 0:1/4:30
const trajclims = (0, 0.04)
const trajalphauni = 6000

#const ylimsdens=[(0,0.2), (0,0.2), (0,0.6), (0,1.2), (0,0.1)]
const ylimsdens=[(0,0.2)]
const ylimstraj=(0,500)

const postcolor = :dodgerblue
const datacolor = :dodgerblue

const kdenpoints = 300

const mplegamma = 0.90
const inverseweightsstd = 20

isp2 = false
col = 1

### individual plot functions
# geandert: nsamples, niter
papersave() = (srand(1); paperplot(); savefig("papergync.pdf"))

test() = (srand(1); paperplot(nsamples=50, niter=20, h=1, zmult=5, smoothmult=5))

global lastresult

function gendata(nsamples, zmult, smoothmult)
  m    = gyncmodel(samplepi1(nsamples), zmult=zmult)
  ms   = smoothedmodel(m, smoothmult)
  muni = gyncmodel(vcat(samplepi0(nsamples), m.xs), zmult=0)
  m, ms, muni
end

function computeweights(m, ms, muni, niter, mplegamma, h)
  w0 = uniformweights(m.xs)
  ws = Dict{String, Vector{Vector{Float64}}}()

  winv = inverseweights(muni.xs)

  println("computing npmle")
  @time ws["NPMLE"] = GynC.em(m, w0, niter)

  println("computing dsmle")
  @time ws["DS-MLE"] = GynC.em(ms, w0, niter);

  println("computing mple")
  @time begin
    ws["MPLE"]  = GynC.mple(m, w0, round(Int,niter/2), mplegamma, h)
    ws["MPLE"]  = vcat(ws["MPLE"], GynC.mple(m, ws["MPLE"][end], round(Int,niter/2), mplegamma, h/10))
  end

  info("max(delta w)", maximum(abs(ws["MPLE"][end] - ws["MPLE"][end-1])))
  ws["uni"] = [winv]
  ws
end

function paperplot(; nsamples=600, zmult=100, smoothmult=200, niter=5000, h=0.01, kwargs...)
  m, ms, muni = gendata(nsamples, zmult, smoothmult)
  ws = computeweights(m, ms, muni, niter, mplegamma, h)
  p = paperplot(m, muni, ws; kwargs...)
  global lastresult = ((m,ms,muni), ws, p)
  p
end

function plotlast()
  ((m, ms, muni), ws, p) = lastresult
  paperplot(m, muni, ws)
end

function paperplot(m, muni, ws; kwargs...)
  isp2=true
  pi0plot = plotrow(ws["uni"], muni; kwargs...)

  let ys = pi0plot[1].series_list[1][:y]
    pi0plot[1].series_list[1][:y] = fill(mean(ys), length(ys))
  end

  aplots = vcat(map(x->plotrow(ws[x], m; kwargs...), ["NPMLE", "DS-MLE", "MPLE"])...)

  plot(pi0plot..., aplots..., size=(1200, 300*3), layout = (4, length(pi0plot)))
end


" return the plots for one row "
function plotrow(ws, m)
  meas = [datas[patient]]


  wpost = bayesposterior(m, meas, ws[end])

  plots = [begin
	     xs = map(x->x[s], m.xs)
	     xlims = (0, GynC.refparms[s] * 5)
	     plotkdeiters(xs, [ws[end]], ylims = ylimsdens[i])
	     plotkde!(xs, wpost, ylims = ylimsdens[i], seriescolor=postcolor, xlims=xlims)
	     end for (i,s) in enumerate(densspecies)]

  plottrajdens(m.xs, ws[end], trajalpha = 16)
  push!(plots, plotdatas!(datas, ylims=ylimstraj, markerstrokecolor=denscolor, color=denscolor, ms=2))
  isp2 = false

  plottrajdens(m.xs, wpost, trajalpha = 8)
  push!(plots, plotdatas!(meas, ylims=ylimstraj, ms=3.5))
  plots
end

### plot helper functions



" plot the kde of iterations of w "
function plotkdeiters(xs, ws; kwargs...)
  colors = (colormap("blues", length(ws)+1)[2:end])'
  p = plot(legend=false; kwargs...)
  for (w,c) in zip(ws, colors)
    c = denscolor 
    plotkde!(xs, w; seriescolor = c)
  end
  p
end


function plotkde!(xs, w; kwargs...)
  #bw = KernelDensity.default_bandwidth(xs) * denskdebwmult
  #@show bw
  k = kde(xs, weights=w, bandwidth=denskdebw, npoints=kdenpoints)
  plot!(k.x, k.density; kwargs...)
end


function trajs(xs)
  hcat([GynC.forwardsol(x, trajts)[:,GynC.measuredinds[trajspecies]] for x in xs]...)
end

" plot the kde of the trajectories "
function plottrajdens(xs::Vector, weights::Vector = uniformweights(xs);
		      tjs = trajs(xs), trajalpha=5, kwargs...)


  #=
  bnd = ylimstraj == :auto ? extrema(trajs) : ylimstraj
  (l,h) = bnd

  bndaugmentation = 0.05
  bndaug = (l-(h-l)*bndaugmentation, h+(h-l)*bndaugmentation)

  kdes = [KernelDensity.kde(filter(x->!isnan(x),trajs[t,:]), boundary = bndaug, weights=weights, npoints = round(Int, kdenpoints*(1+2*bndaugmentation)), bandwidth=10) for t in 1:size(trajs, 1)]

  inds = find(x->(x>=l&&x<=h), kdes[1].x)

  ys = kdes[1].x[inds]
  dens = hcat([k.density[inds] for k in kdes]...)

  #clims = (0, quantile(vec(dens), cquant))
  #clims = (0, maximum(dens) * cquant)
  #println("maximal traj density: $(maximum(dens)); 98% quantile: $(quantile(vec(dens), 0.98))")

  p=contour(trajts, ys, dens, clims=trajclims, fill=true, seriescolor = :heat, legend=false, kwargs...)

  =#

  p=plot(legend=false, ylims=ylimstraj)
  @show maximum(weights)
  for i in 1:size(tjs, 2)
    a = isp2 ? trajalphauni : trajalpha
    a = min(1., weights[i]* a)
    plot!(p, trajts, tjs[:,i], alpha=a, color=:black)
  end
  p
end

" plot the given data "
function plotdatas!(datas; kwargs...)
  specdatas = map(d->d[:,trajspecies], datas)
  scatter!(0:30, specdatas, color=datacolor, markerstrokecolor=datacolor, legend=false, ms=1.5; kwargs...)
end


### model generation

" generate a gync likelihoodmodel "
function gyncmodel(xs; zmult = 0)
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
    !any(isnan(GynC.forwardsol(x, trajts))) && push!(xs, x)
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
function inverseweights(xs::Vector)
  w=1./mykde(xs,xs,inverseweightsstd)
  normalize!(w, 1)
end

function uniformweights(xs::Vector)
  ones(length(xs)) / length(xs)
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

