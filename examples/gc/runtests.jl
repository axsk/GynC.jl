info("testing sample()")
s =  GynC.sample(Config(), 100)
@time s = GynC.sample!(s, 100)

#=using Plots
unicodeplots()
display(plot(s))=#

info("testing batch sampling")
tmp = tempname() * ".jld"
s = batch(tmp, batchiters=10, maxiters=20, config=Config(Lausanne(1), thin=2))
s = load(tmp)

@assert size(s.samples) == (10, 115)

rm(tmp)
