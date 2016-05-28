info("testing simulate()")
GynC.simulate(Config(), 100)

info("testing mcmc")
sim = mcmc(Config(Lausanne(1)), 10)

info("testing batch sampling")
tmp = tempname() * ".jld"
s = batch(tmp, batchiters=10, maxiters=20, config=Config(Lausanne(1), thin=2))
s = load(tmp)


@assert size(s.samples) == (10, 115)

rm(tmp)
