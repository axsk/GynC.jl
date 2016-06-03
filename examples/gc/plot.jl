using Plots
import Plots: plot

import GynC: Sampling

gauge(m::Matrix) = m ./ m[1, :]
gauge(v::Vector) = v / v[1]

function plot(s::Sampling)
  Plots.plot(s.samples |> gauge)
end

relpost(s::Sampling) = exp(s.logpost - s.logpost[1])

velocity(m::Matrix) = m[1:end-1,:] - m[2:end,:]
normalize(m::Matrix) = m ./ mean(m, 1)

function transform(s::Sampling, f::Function)
  s = deepcopy(s)
  s.samples = f(s.samples)
end

plot_t_llh(sim) = plot(sim.loglikelihood, title="loglikelihood")

samples(s::Sampling) = s.samples

measuredspeciesname(species) = speciesnames[measuredinds[species]]

function plotdata(s::Sampling, species;
  p=plot(), title=measuredspeciesname(species), kwargs...)

  y = data(s)[:, species] |> vec
  x = find(yy -> !isnan(yy), y)

  plot!(p, x-1, y[x]; title=title, kwargs...)
end

function plotsolutions(s::Sampling, species;
  p=plot(), 
  x=0:1/3:30, 
  title=measuredspeciesname(species),
  kwargs...)

  data = hcat(Vector{Float64}[s[:, measuredinds[species]] for s in solutions(s, x)]...)
  plot!(p, x, data; title=title, kwargs...)
  p
end
