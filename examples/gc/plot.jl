using Plots
import Plots: plot

import GynC: Sampling

gauge(m::Matrix) = m ./ m[1, :]
gauge(v::Vector) = v / v[1]

function plot(s::Sampling)
  Plots.plot(s.samples |> gauge)
end

relpost(s::Sampling) = exp(s.logpost - s.logpost[1])

import Base.getindex
function Base.getindex(s::GynC.Sampling, a, b)
  GynC.Sampling(
    s.samples[a,b], 
    s.logprior[a], 
    s.logpost[a], 
    s.config,
    s.variate,
    s.thin)
end

velocity(m::Matrix) = m[1:end-1,:] - m[2:end,:]
normalize(m::Matrix) = m ./ mean(m, 1)

function transform(s::Sampling, f::Function)
  s = deepcopy(s)
  s.samples = f(s.samples)
end

plot_t_llh(sim) = plot(sim.loglikelihood, title="loglikelihood")

samples(s::Sampling) = s.samples
