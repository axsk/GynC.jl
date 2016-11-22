using GynC
using JLD
using KernelDensity
using Plots

@everywhere begin
	import Distributions: pdf, rand
	type MatrixNormalCentered <: Distribution
		sigmas
	end

	function rand(n::MatrixNormalCentered)
		map(s->rand(Normal(0,s)), n.sigmas)
	end

	function pdf(n::MatrixNormalCentered, x::Matrix)
		@assert size(n.sigmas) == size(x)
		d = 0.
		for i in length(x)
			isnan(x[i]) && continue
			d += logpdf(Normal(0, n.sigmas[i]), x[i])
		end
		exp(d)
	end
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

function sampleuniform(nsamples)
	yprior = GynC.priory0(1)
	xsuni = Vector{Float64}[]
	for i = 1:10000
		trial = vcat(GynC.refparms.* rand(82), rand(yprior))
		!any(isnan(GynC.forwardsol(trial))) && push!(xsuni, trial)
		length(xsuni) == 1000 && break
	end
	xsuni
end

function alldatas(; minmeas=10)
	datas = vcat([Lausanne(i).data for i=1:40])#, [GynC.Pfizer(i).data for i=1:13])
	datas = filter(d->length(d) - sum(isnan(d)) > 20, datas)
end

function plotkdeiters(xs, ws; bandwidthmult=0.1, kwargs...)
	colors = itercolors(length(ws))

	plot(legend=false, xlims=xlimsdens; kwargs...)

	for i in 1:length(ws)
		bw = KernelDensity.default_bandwidth(xs)
		k=kde(xs, weights=ws[i], bandwidth=bw*bandwidthmult)
		plot!(k.x, k.density, seriescolor = colors[i], ylims = ylimsdens)#, linecolor = colors[i])
	end

	plot!()
end
#=
nicespecies = [31 44 50 76]

spec = 44
for spec in nicespecies
	for (k,v) in ws
		v=v[1:20:end]
		plotkdeiters(map(x->x[spec], xs), v; title="$spec $(GynC.samplednames[spec])  $k", ylims = (0,0.3)) |> display

		#plot(plot(map(w->GynC.hz(m,w), v), title="hz $k"), plot(map(w->GynC.logl(m,w), v), title="logl $k")) |> display
	end
end
=#


function plottrajdens(ts, sols, w::Vector; quant = 0.99)
	msol = hcat(sols...)
	#bnd = (-1, quantile(vec(msol), quant) * 1.01)
	bnd = ylimstraj


	kdes = [KernelDensity.kde(filter(x->!isnan(x),msol[t,:]), boundary = bnd, weights=w) for t in 1:size(msol, 1)]
	ys = kdes[1].x

	dens = hcat([k.density for k in kdes]...)
	clims = (0, quantile(vec(dens),0.98))

	contour(ts, ys, hcat([k.density for k in kdes]...), clims=clims, fill=true, seriescolor = :heat, legend=false)
end


function plottraj(ts, sols, w; yquantile = 0.99, alphamult = 10)
	ylims = (0, quantile(vcat(sols...), 0.99))
	plot(ts, sols, alpha=w'*alphamult, legend=false, ylims=ylims, color = :black, linewidth=3)
end

function plotdatas!(datas, species = 3)
	specdatas = map(d->d[:,species], datas)
	scatter!(0:30, specdatas, color=:blue, legend=false, ms=1, ylims = ylimstraj)
end

#plottraj(ts, sols, wuni)
#plotdatas!(datas)

function bayesposterior(m, data, wprior)
	L = likelihoodmat(m.ys, data, m.measerr)
	GynC.emiteration(wprior, L)
end

#patientdata = [datas[3]]

#plottraj(ts, sols, bayesposterior(m, patientdata, wuni))
#plotdatas!(patientdata)

function plotcol(ws, xs=xs) 
	meas = [datas[3]]
	ylims = (0,0.03)
	densspecies = 8

	ts = 0:1/4:30
	trajspecies = GynC.measuredinds[3]
	sols = [GynC.forwardsol(x, ts)[:,trajspecies] for x in xs];

	xs = map(x->x[densspecies], xs)
	pprior = plotkdeiters(xs, ws, bandwidthmult = 0.6)

	pprioq = plottrajdens(ts, sols, ws[end], quant=0.95)
	plotdatas!(datas)

	ppostq = plottrajdens(ts, sols, bayesposterior(m, meas, ws[end]))
	plotdatas!(meas)
	[pprior, pprioq, ppostq]
end

###  load samples

@time s = JLD.load("../data/0911/allsamples.jld")["samples"]
xs = subsample(s, 1000, 100_000);
@show length(xs);

datas = alldatas();
@show length(datas);

phi(x) = GynC.forwardsol(x)[:,GynC.measuredinds]
@time ys = phi.(xs);

naninds = find(x->any(isnan(x)), ys)
nonnaninds = deleteat!(collect(1:length(ys)), naninds)

xs = xs[nonnaninds]
ys = ys[nonnaninds];

#xs0 = sampleuniform(length(xs))

err = let sigma = 0.1
	MatrixNormalCentered(repmat(sigma*GynC.model_measerrors', 31))
end

zmult = 10
zs = map(y->y+rand(err), repmat(ys, zmult));
@show length(zs)


### create models

m = GynC.LikelihoodModel(xs, ys, zs, datas, err);

sigmas = [1*KernelDensity.default_bandwidth(filter(x->!isnan(x),[d[i,j] for d in datas])) for i=1:31, j=1:4]
smoothkernel = MatrixNormalCentered(sigmas)

ms = GynC.smoothdata(m, 200, smoothkernel);

sigmanew = sqrt.(m.measerr.sigmas .^ 2 + smoothkernel.sigmas .^ 2)
ms.measerr = MatrixNormalCentered(sigmanew)


# estimate priors

niter = 200
h = 5

wuni = ones(length(xs)) / length(xs);
w0 = wuni

ws = Dict{String, Vector{Vector{Float64}}}()
@time ws["NPMLE"] = GynC.em(m, w0, niter)
@time ws["DS-MLE"] = GynC.em(ms, w0, niter);
@time ws["MPLE"]  = GynC.mple(m, w0, niter, 0.9, h)
@time ws["Reference Prior"] = GynC.mple(m, w0, niter, 1, h);

### plot results

itercolors(n) = collect(colormap("blues", n))'

xlimsdens = (0, 25)
ylimsdens = (0, 0.2)

ylimstraj = (-1, 580)

aplots = vcat(map(x->plotcol(ws[x]), ["NPMLE", "DS-MLE", "MPLE"])...)

#uniplots = plotcol([wuni, wuni, wuni], xs0)
#uniplots[1] = plot(0:25, x->1/25, ylims=ylimsdens, xlims = xlimsdens, legend=false)

#aplots = vcat(uniplots, aplots) 

plot(aplots..., size=(1200, 300*3), layout = (Int(length(aplots)/3), 3))


### invert pi1 to pi0

function uniformweights(xs::Vector, stdmult=30)
	w=1./mykde(xs,xs,stdmult)
	normalize!(w, 1)
end


function mykde(data, evals, stdmult)
	dim = length(data[1])
	stds = [KernelDensity.default_bandwidth(map(x->x[d], data)) for d in 1:dim] * stdmult
  map(evals) do e
		pdf(MvNormal(e, stds), hcat(data...)) |> sum
	end
end

winv = uniformweights(xs)

plotcol([winv, winv], xs)
