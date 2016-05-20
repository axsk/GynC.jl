tmp = tempname() * ".jld"

s = batch(tmp, batchiters=10, maxiters=20, thin=2, config=ModelConfig(Lausanne(1)))
s = load(tmp, all=true)

@assert size(s.samples) == (10, 116, 1)

rm(tmp)
