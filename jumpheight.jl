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
  cc = []
  for yi in y
    c = posterior(yi)
    if cc == []
      cc = c
    else
      cc.value = cat(1, cc.value, c.value)
    end
  end
  cc
end

""" Given some measured y value, sample the posterior distribution. """
function posterior(y::Real; iters=1000)
  m = Model(
    x = Stochastic(()->DensityDistribution(x->rho(y-h(x)))))

  setsamplers!(m, [Slice([:x],[100.0])])
  data =  Dict{Symbol,Any}()
  init = [Dict{Symbol,Any}(:x => 60) for i=1:3]

  mcmc(m, data, init, iters, burnin=0, thin=1, chains=3)
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

	mcmc(m, data, init, 10000, burnin=0, thin=1, chains=3)
end
