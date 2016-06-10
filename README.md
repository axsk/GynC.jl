# GynC.jl
[![Build Status](https://travis-ci.org/axsk/GynC.jl.svg?branch=master)](https://travis-ci.org/axsk/GynC.jl) [![codecov.io](https://codecov.io/github/axsk/GynC.jl/coverage.svg?branch=master)](https://codecov.io/github/axsk/GynC.jl?branch=master)


This package provides the toolchain to estimate the parameters of the GynCycle model.

Its code mainly revolves around the three main types:

- `Config` which stores the bayesian model configuration, i.e. the patient data, measurement error, priors, ... as well as the MCMC proposal density and the thinning.
- `Sampling` which is obtained by sampling from a config via `sample(::Config, iters)` containing the sampled points, the current sampler state, as well as a reference to its initial config.
- `WeightedChain` which can be constructed from multiple samplings via `WeightedChain(::Vector{Sampling})` representing the average/merged chain and is used for further prior estimation steps, for example the EM-iteration: `emiteration!(::WeightedChain)`
