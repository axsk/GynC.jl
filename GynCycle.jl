module Jumpheight

gauss(x,m,s)=1/(s*sqrt(2*pi))*exp(-1/2*((x-m)/s)^2)
gauss(x) = gauss(x,0,10)

h(x)=2700*x/470*(1-x/470)^5
y=[50,72]

sims = [ begin
	model = Model(
		x = Stochastic(Uniform(0,100)) # prior for the weight
		gauss = Stochastic(x->gauss(yi-h(x))) ) 
	setsampler!(model, NUTS(:gauss))
	inits = 0.5 
	mcmc(model)
	end for yi = y]
end

""" estimate prior 
function priorestimate(sims)
	sum up each sim for l1 integral
	"weight" the result. TODO: HOW
end

not realy sampling the prior now...
alternative: intrude into model: """

#y = Stochastic(Uniform(y))
#find way to store integral 
