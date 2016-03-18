#= using KernelDensity 
using PyPlot

function plotprior(species, factor=5; kwargs...)
  if species > 82 
    prior = Gync.independentmixtureprior()[species-82]
    min, max = PyPlot.xlim()
    x=linspace(min, max, 200)
    PyPlot.plot(x, pdf(prior, x); kwargs...)
  else
    min = 0
    max = Gync.refparms[species] * factor
    val = 1/(max-min)
    PyPlot.plot([min,max], [val,val]; kwargs...)
  end
end

function plotchain(data, species)
  
end

function plotchainspecies(data::Vector)
  npoints = 2048
  bw = maximum(data) / npoints * 5
  k = kde(data, npoints=2048, bandwidth=bw)
  plot(k.x, k.density)
end
=#

using Vega
import KernelDensity

"return the marginal density of a given species"
function plot_species(c::Mamba.AbstractChains; kde=false, nbins=40)
  @assert size(c,[2,3]) == (1,1)
  data = c.value |> vec
  
  v = kde ?
    plot_kde(data):
    histogram(x=data, nbins = nbins)
  
  title!(v, title=c.names[1])
end

"kdeplot using the KernelDensity package"
function plot_kde(data::Vector; npoints = 2^14, bandwidth = maximum(data) / npoints * 5)
  k = KernelDensity.kde(data, npoints=npoints, bandwidth = bandwidth)
  v = lineplot(x = k.x, y = k.density)
  # v.scales[1].zero = false
end

"returns slices like in mapslices, but works on general types supporting size(), getindex()" 
function slices(a,dim::Int)
  inds = Any[(:) for i in 1:length(size(a))]
  map(1:size(a,dim)) do i
    inds[dim] = i
    getindex(a,inds...)
  end
end

"dispatch multiple species to their marginal density plots"
function plot(c::Mamba.AbstractChains)
  reduce(layer, map(plot_species, slices(c, 2)))
end

### residuum plots

import PyCall
import PyPlot
PyCall.@pyimport seaborn as sns

function plot_residuum(c::Mamba.ModelChains, species::Int; t = 0:0.1:30, kwargs...)
  PyPlot.figure()
  speciesid = measuredinds[species]
  sols = samplesolutions(c,t)
  cols = sns.dark_palette("seagreen", n_colors=length(sols))
  for (i,sol) in enumerate(sols)
    PyPlot.plot(collect(t), sol[speciesid,:] |> vec, color=cols[i], zorder=1; kwargs...)
  end
  PyPlot.plot(collect(0:30.), c.model[:data][species,:] |> vec, "o", zorder=2)
end

function samplesolutions(c::Mamba.ModelChains, t)
  map(1:size(c.value,1)) do i
    parms = allparms(c[:,:parms,:].value[i,:,:] |> vec)
    y0    = c[:,:y0,:].value[i,:,:] |> vec
    gync(y0, t, parms)
  end
end
