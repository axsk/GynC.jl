info("testing sample()")
s =  GynC.sample(Config(), 100)
@time s = GynC.sample!(s, 100)

#=using Plots
unicodeplots()
display(plot(s))=#

info("testing batch sampling")
tmp = tempname() * ".jld"
s = batch(Config(Lausanne(1), thin=2), [10,20], tmp)
s = load(tmp)
s = GynC.sample!(s, 20)
@assert size(s.samples) == (20, 116)
rm(tmp)


info("testing sampling plots")
plotsolutions(s, 1)
plotdata(s, 1)


info("testing weightedchain")
w=WeightedChain([s, s])
GynC.sample(w, 10)

testslurm = false
if testslurm
  info("testing batch")
  dir = mktempdir()
  cs = [Config(Lausanne(i)) for i in 1:3]
  ss = batch(cs, [10], dir=dir)
  @assert typeof(ss[3]) == GynC.Sampling
  rm(dir, recursive=true)
end
