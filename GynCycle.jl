module Jumpheight

using Mamba, Distributions
include("utils.jl")

g(x,m,s)=1/(s*sqrt(2*pi))*exp(-1/2*((x-m)/s).^2)
g(x) = g(x,0,10)

h(x)=2700*x/470*(1-x/470)^5
y=[50,72]
yi=y[1]
model = Model(
	x = Stochastic(()->Uniform(0,100)), # prior for the weight
	gauss = Stochastic(x->PDFDistribution(xx->g(y-h(xx)))))
setsamplers!(model, [NUTS([:gauss])])
data =  Dict{Symbol,Any}(:y => 72)
init = [Dict{Symbol,Any}(:x => 0.5, :gauss => 50) for i=1:2]
res=mcmc(model, data, init, 1000, burnin=200, thin=2, chains=2)

end



#= estimate prior  
function priorestimate(sims)
	sum up each sim for l1 integral
	"weight" the result. TODO: HOW
end

not realy sampling the prior now...
alternative: intrude into model: """

#y = Stochastic(Uniform(y))
#find way to store integral 
=#
