using Plots: plot, plot!, scatter!
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

  scatter!(p, x-1, y[x]; title=title, kwargs...)
end

### Plots recipces

# TODO:
# - plotdata 
# - plotdata with period
# - plot reference

using Plots

@userplot PlotSolutions
@recipe function f(o::PlotSolutions)
  local samples

  if isa(o.args[1], Matrix) 
    samples = o.args[1]
  elseif isa(o.args[1], Sampling) 
    samples = o.args[1].samples
  elseif isa(o.args[1], WeightedChain)
    samples = sample(o.args[1], o.args[2])
  else
    error("no valid samples given")
  end

  species       --> measuredinds
  t             --> 0:1/3:30
  color_palette --> [colorant"steelblue"]
  linewidth     --> 0.1
  label         --> ""
  seriesalpha   --> 0.1
  layout        --> length(d[:species])

  species  = pop!(d, :species)
  t        = pop!(d, :t)

  nspecies = length(species)
  nsamples = size(samples, 1)

  subplot := repeat(collect(1:nspecies)', outer=[1, nsamples])

  solutions = hcat([gync(samples[i,:] |> vec, t)[:,species] for i in 1:nsamples]...)

  t, solutions
end




#=
@require PyPlot begin

  function plot(s::WeightedChain, species, nbins = 20)
    x,y = weightedhist(s.samples[:,species], s.weights, nbins)
    PyPlot.bar(x[1:end-1], y, width=step(x))
  end
end
=#

import KernelDensity: kde

kde(s::WeightedChain, species, npoints) = kde(s.samples[:,species], npoints = npoints, weights = s.weights)

function plot(s::WeightedChain, species, npoints)
  k = kde(s.samples[:,species], npoints = npoints, weights = s.weights)
  plot(k.x,k.density)
end

function weightedhist(v::Vector, w::Vector, nbins)
  min, max = extrema(v)
  step = (max-min) / nbins

  bins = zeros(nbins)

  for i in eachindex(v)
    bin = round(Int, (v[i] - min) / step, RoundDown) + 1
    if bin > nbins 
      bin = nbins
    end
    bins[bin] += w[i]
  end
  (min:step:max), bins
end
