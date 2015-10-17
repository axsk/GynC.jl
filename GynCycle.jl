using Mamba, Distributions
include("utils.jl")

g(x,m,s)=1/(s*sqrt(2*pi))*exp(-1/2*((x-m)/s).^2)
rho(x) = g(x,0,10)
prior(x) = (1/8*(x/40)^2*(1-x/40)^2*(x>0)*(x<40)*30/40
	+ 7/8*g(x,78,21))
h(x)=2700*x/470*(1-x/470)^5

y=[150,130] # samples
yi=y[1]

""" Given jumpheights of different persons, return samples of the estimated prior (mean of the posteriors). """
function priorestimation(y::Vector)
	res = []
	samples = [	begin
			m = Model(
				x = Stochastic(()->DensityDistribution(x->rho(yi-h(x)))))

			setsamplers!(m, [Slice([:x],[100.0])])
			data =  Dict{Symbol,Any}()
			init = [Dict{Symbol,Any}(:x => 60) for i=1:3]

			# cat along right dimension
			mcmc(m, data, init, 5000, burnin=0, thin=1, chains=3)
		end for yi in y]
	return samples
end

""" Given a prior (weight) and likelihood (height|weight) sample the posterior (height). This simulates measurements across the population."""
function sampleweights()
	m = Model(
		x = Stochastic(()->DensityDistribution(x->prior(x))),
		y = Stochastic(x ->DensityDistribution(y->rho(y-h(x)))))

	setsamplers!(m, [Slice([:x,:y],[100,100])])
	setsamplers!(m, [Slice([:x],[1000]), Slice([:y],[1000])])
	setsamplers!(m, [NUTS([:x,:y])])
	data =  Dict{Symbol,Any}()
	init = [Dict{Symbol,Any}(:x => 60, :y => 100) for i=1:3]

	mcmc(m, data, init, 100000, burnin=0, thin=1, chains=3)
end

sim=sampleweights()
