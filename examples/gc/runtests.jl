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
s = sample!(s, 20)
@assert size(s.samples) == (20, 115)
rm(tmp)


info("testing sampling plots")
Plots.unicodeplots()
plotsolutions(s, 1)
plotdata(s, 1)


info("testing weightedchain")
w=WeightedChain(s, s)
sample(w, 10)
