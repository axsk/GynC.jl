tmp = tempname() * ".jld"

s = batch(tmp, batchiters=10, maxiters=20, config=GynCConfig(Lausanne(1)))
s = load(tmp)

@assert size(s.samples) == (10, 115, 1)

rm(tmp)
