using GynC

sim = mcmc(ModelConfig(Lausanne(1)), 100)

mktemp() do file
  batch(file, batchiters=10, maxiters=100, thin=2, config=ModelConfig(Subject(:lausanne, 1)))
end

