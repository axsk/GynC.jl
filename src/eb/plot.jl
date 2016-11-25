using KernelDensity
using Plots


function unfoldcols{T}(v::Vector{Vector{T}})
  x = map(e->e[1], v)
  y = map(e->e[2], v)
  x,y
end

function plot_kde{T}(zs::Vector{Vector{T}}, wz::Vector; bandwidth=5., kwargs...)
  k=kde(unfoldcols(zs), weights=wz, bandwidth=(bandwidth,bandwidth))
  contour(k.x, k.y, reshape(k.density, length(k.x), length(k.y))'; kwargs...)
end

function plotiters(xs, ws; kwargs...)
  color = collect(colormap("blues", length(ws)))'
  plot(xs, ws, legend=false, seriescolor=color)
end
