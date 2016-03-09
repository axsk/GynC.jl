using GynC

sim = mcmc(ModelConfig(Lausanne(1)), 100)

tmp = tempname()
sim = batch(tmp, batchiters=10, maxiters=20, thin=2, config=ModelConfig(Lausanne(1)))
@assert step(sim) == 2
@assert last(sim) == 20
rm(tmp)
