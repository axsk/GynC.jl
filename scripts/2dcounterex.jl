using GynC, Plots, LaTeXStrings
import Plots.scatter

pyplot(grid=false)

include("../src/eb/plot.jl")

function scatter(ys::Vector{Vector{Float64}}; kwargs...)
  m = hcat(ys...)
  scatter(m[1,:], m[2,:]; kwargs...)
end



function paperplot(;
  sigma = 1.5,
  nx = 200,
  nz = 300,
  niter = 1000,
  h = 1,
  ms = 2,
  dlims = (0,0.03),
  lims = (-14,14),
  clims = (0,0.004),
  kwargs...)

  m = Federn.federmodel(nx, 0, nz, sigma, xmax=50)
  w0 = Federn.wbeta(m.xs, maximum(m.xs));
  wuni = normalize!(ones(length(m.xs)),1)

  wzs = GynC.mple(m, w0, niter, 1, h)
  wz = wzs[end];

  plot(map(norm, wzs[2:end]-wzs[1:end-1]), title="d/dt |wz(t)|_2")

  msscale = length(m.ys) * ms
  
  p = []

  labels = [[L"$w_0$", L"$y_k \ \mathrm{with} \  w_0  \ \mathrm{weights}$", L"$\rho_Z(z \mid W=w_0)$"], [L"$w_{\rm ref}$", L"$y_k \ \mathrm{with} \ w_{\rm ref} \ \mathrm{weights}$", L"$\rho_Z(z \mid W=w_{\rm ref})$"]]


  for (w,l) in zip([wuni, wz], labels)
    push!(p, plot(m.xs, w, label=l[1], ylims=dlims, linewidth=1, color=:dodgerblue, yticks=false; kwargs...))

    tmp = scatter([[NaN,NaN]], label=l[2])

    push!(p, scatter!(tmp, unfoldcols(m.ys)..., ms=w*msscale, lims=lims, label="", color=:dodgerblue; kwargs...))
    push!(p, plot_kde(m.ys, w, bandwidth=sigma, lims=lims, clims=clims, cbar=false, label=l[3], legend=true; kwargs...))
    plot!(p[end], [1],[1], ms=0, label=l[3])
  end



  plot(p..., layout=(2,3))#, legendfont = font(18))
end
  


