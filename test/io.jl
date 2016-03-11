tmp = tempname()

sim = batch(tmp, batchiters=10, maxiters=20, thin=2, config=ModelConfig(Lausanne(1)))
sim = load(tmp, all=true)

@assert step(sim) == 2
@assert last(sim) == 20

@assert size(sim.value) == (10, 116, 1)

rm(tmp)
