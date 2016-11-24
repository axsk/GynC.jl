using GynC
using Plots, Distributions

pyplot(reuse=false)

xs = 1:1/3:50
phi(k) = (GynC.Federn.odeohnetreatment(k))[1] # only 1 measurement
plot(xs, phi.(xs))

function syntethiclikelihoodmodel(xs, phi, prior::Distribution, ndata, zmult, measerr)
	ys = phi.(xs)
	zs = repmat(ys, zmult) + rand(measerr, length(ys)*zmult)
	datas = phi.(rand(prior, ndata)) + rand(measerr, ndata)
	GynC.LikelihoodModel(xs, ys, zs, datas, measerr)
end

function dslikelihoodmodel(m::GynC.LikelihoodModel, dmult, sigmak)
	sdatas = repmat(m.datas, dmult) + rand(Normal(0,sigmak), length(m.datas) * dmult);
	smeaserr = Normal(0, sqrt(m.measerr.σ^2 + sigmak^2))
	ms = GynC.LikelihoodModel(m.xs, m.ys, m.zs, sdatas, smeaserr, m.zsampledistr)
end

measerr = Normal(0,2)
ndata = 300
augz = 100
prior = GynC.Federn.prior

m = syntethiclikelihoodmodel(xs, phi, prior, ndata, augz, measerr)

augd = 100
stdd = KernelDensity.default_bandwidth(m.datas)
ms = dslikelihoodmodel(m, augd, stdd);

w0 = ones(length(xs)) / length(xs);
wprior = pdf(prior, xs)
wprior = wprior / sum(wprior);

niter = 400
h = 0.1

ws = Dict()
@time ws["NPMLE"] = GynC.em(m, w0, niter)
@time ws["DS-MLE"] = GynC.em(ms, w0, niter);
@time ws["MPLE"]  = GynC.mple(m, w0, niter, .9, h)
@time ws["Reference Prior"] = GynC.mple(m, w0, niter, 1, h)

labels = ["NPMLE" "DS-MLE" "Reference Prior" "MPLE"]
densities = map(l->ws[l][end], labels) |> vec
plot(xs, wprior, label="True Prior", linewidth=2, legendfont=font(10), tickfont=font(8), ylims=(0,0.05), size=(600, 350), grid=false)
plot!(xs, densities, labels=labels, linewidth=1.2)

plot(xs, GynC.Federn.maxamplitude)
