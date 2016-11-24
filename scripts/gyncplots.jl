using GynC
using JLD
using KernelDensity
using Plots

import Distributions: pdf, rand, logpdf
type MatrixNormalCentered <: Distribution
  sigmas
end

function rand(n::MatrixNormalCentered)
  map(s->rand(Normal(0,s)), n.sigmas)
end

function pdf(n::MatrixNormalCentered, x::Matrix)
  @assert size(n.sigmas) == size(x)
  d = 0.
  for i in 1:length(x)
    isnan(x[i]) && continue
    d += logpdf(Normal(0, n.sigmas[i]), x[i])
  end
  exp(d)
end

function logpdf(n::MatrixNormalCentered, x::Matrix)
  @assert size(n.sigmas) == size(x)
  d = 0.
  for i in 1:length(x)
    isnan(x[i]) && continue
    d += logpdf(Normal(0, n.sigmas[i]), x[i])
  end
  d
end

function mlikelihoodmat(zs, ys, d)
  x= Array(Float64, (length(zs), length(ys)))
  for j=1:length(ys)
    for i=1:length(zs)
      x[i,j] = logpdf(d, zs[i]-zs[j])
    end
  end
  exp(x-maximum(x))
end

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

function samplepi1(nsamples)
  yprior = GynC.priory0(1)
  xs = Vector{Float64}[]
  while length(xs) < nsamples
    x = vcat(GynC.refparms.* rand(82) * 5, rand(yprior), 30)
    !any(isnan(GynC.forwardsol(x))) && push!(xs, x)
  end
  xs
end

function alldatas(; minmeas=20)
  datas = vcat([Lausanne(i).data for i=1:40])#, [GynC.Pfizer(i).data for i=1:13])
  datas = filter(d->length(d) - sum(isnan(d)) > minmeas, datas)
end

function plotkdeiters(xs, ws; bandwidthmult=0.1, kwargs...)
  colors = colormap("blues", length(ws)+1)[2:end]'

  plot(legend=false; kwargs...)

  for (w,c) in zip(ws, colors)
    bw = KernelDensity.default_bandwidth(xs)
    k=kde(xs, weights=w, bandwidth=bw*bandwidthmult)
    plot!(k.x, k.density, seriescolor = c)#, linecolor = colors[i])
  end

  plot!()
end

# nicespecies = [31 44 50 76]


function plottrajdens(ts, sols, w::Vector; ylims = :auto, kwargs...)
  msol = hcat(sols...)

  bnd = ylims == :auto ? extrema(msol) : ylims

  kdes = [KernelDensity.kde(filter(x->!isnan(x),msol[t,:]), boundary = bnd, weights=w) for t in 1:size(msol, 1)]

  ys = kdes[1].x
  dens = hcat([k.density for k in kdes]...)

  clims = (0, quantile(vec(dens), 0.98))

  contour(ts, ys, dens, clims=clims, fill=true, seriescolor = :heat, legend=false, kwargs...)
end


#=function plottraj(ts, sols, w; yquantile = 0.99, alphamult = 10)
  ylims = (0, quantile(vcat(sols...), 0.99))
  plot(ts, sols, alpha=w'*alphamult, legend=false, ylims=ylims, color = :black, linewidth=3)
end=#

function plotdatas!(datas, species = 3; kwargs...)
  specdatas = map(d->d[:,species], datas)
  scatter!(0:30, specdatas, color=:blue, legend=false, ms=1; kwargs...)
end

#plottraj(ts, sols, wuni)
#plotdatas!(datas)

function bayesposterior(m, data, wprior)
  L = mlikelihoodmat(m.ys, data, m.measerr)
  GynC.emiteration(wprior, L)
end

#patientdata = [datas[3]]

#plottraj(ts, sols, bayesposterior(m, patientdata, wuni))
#plotdatas!(patientdata)

function plotrow(ws, m;
  patient = 4,
  ylimsdens = :auto,
  ylimstraj = (0,400),
  densspecies = [8 31 44 50 76]
 )
  
  meas = [datas[patient]]

  ts = 0:1/4:30
  trajspecies = 3
  sols = [GynC.forwardsol(x, ts)[:,GynC.measuredinds[trajspecies]] for x in m.xs];

  plots = Plots.Plot[]

  for s in densspecies
    sxs = map(x->x[s], m.xs)
    push!(plots, plotkdeiters(sxs, ws, bandwidthmult = 0.5, ylims=ylimsdens))
  end

  plottrajdens(ts, sols, ws[end], ylims=ylimstraj)
  push!(plots, plotdatas!(datas, trajspecies, ylims=ylimstraj))

  @show ws[end] - bayesposterior(m, meas, ws[end])

  plottrajdens(ts, sols, bayesposterior(m, meas, ws[end]), ylims = ylimstraj)
  push!(plots, plotdatas!(meas, trajspecies, ylims=ylimstraj))
  plots
end

### calculate weights for any sampling to achieve a uniform distribution

function uniformweights(xs::Vector, stdmult=30)
  w=1./mykde(xs,xs,stdmult)
  normalize!(w, 1)
end

# compute the kde evaluations at 'evals', given the points 'data'.
# for high dimensions adjust stdmult to tweak the covariance
function mykde(data, evals, stdmult)
  dim = length(data[1])
  stds = [KernelDensity.default_bandwidth(map(x->x[d], data)) for d in 1:dim] * stdmult
  map(evals) do e
    pdf(MvNormal(e, stds), hcat(data...)) |> sum
  end
end


###  load samples

function samplepi0(n, burnin=100_000)
  s = JLD.load("../data/0911/allsamples.jld")["samples"]
  xs = subsample(s, n, burnin);
end

global datas = alldatas();

function gyncmodel(xs; zmult = 0, sigma=0.1)
  phi(x) = GynC.forwardsol(x)[:,GynC.measuredinds]
  ys = phi.(xs);

  nonaninds = find(x->!any(isnan(x)), ys)

  xs = xs[nonaninds]
  ys = ys[nonaninds]

  err = MatrixNormalCentered(repmat(sigma*GynC.model_measerrors', 31))

  zs = map(y->y+rand(err), repmat(ys, zmult));

  m = GynC.LikelihoodModel(xs, ys, zs, datas, err);
end

function smoothedmodel(m, smoothmult)
  sigmas = [1*KernelDensity.default_bandwidth(filter(x->!isnan(x),[d[i,j] for d in datas])) for i=1:31, j=1:4]
  smoothkernel = MatrixNormalCentered(sigmas)

  ms = GynC.smoothdata(m, smoothmult, smoothkernel);

  sigmanew = sqrt.(m.measerr.sigmas .^ 2 + smoothkernel.sigmas .^ 2)
  ms.measerr = MatrixNormalCentered(sigmanew)
  info("adjusted meas error")

  ms
end

function paperplot(;nsamples = 100, niter=50, h=5, zmult=10, smoothmult=10, kwargs...)
  m    = gyncmodel(samplepi0(nsamples), zmult=zmult)
  ms   = smoothedmodel(m, smoothmult)
  muni = gyncmodel(samplepi1(nsamples), zmult=0)

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
  winv = uniformweights(muni.xs)
  pi0plot = plotrow([winv], muni; kwargs...)
  ncol = length(pi0plot)

  @show typeof(pi0plot), typeof(plotrow(ws["MPLE"], m))

  aplots = vcat(pi0plot, map(x->plotrow(ws[x], m; kwargs...), ["NPMLE", "DS-MLE", "MPLE"])...)
  @show aplots
  plot(aplots..., size=(1200, 300*3), layout = (Int(length(aplots)/ncol), ncol))
end
