
module Feder
import Distributions

include("odes.jl")

const rhoe = Distributions.MvNormal(2,5)

phi = odeohnetreatment

perturb(y, rho_std) = y + rand(MvNormal(2, rho_std))

function generatepriordata(n, rho_std)
    k1 = 15
    k2 = 30
    s1 = 0.15*k1
    s2 = 0.15*k2
    n1 = floor(Int, n/2)
    x1 = k1 + s1*randn(n1)
    x2 = k2 + s2*randn(n-n1)

    xs = [x1; x2]
    zs = perturb.(phi.(xs), rho_std)
end


function federexperiment(;
                         nx = 300, # number of samples in parameter space
                         ndata    = 2000, # number of simulated data points
                         rho_std = 5, # standarddeviation of the meas. error
                         xmin = 1, xmax = 110,
                         zmult = 1) # multiplier for z measurements in hz computation)

    xs = linspace(xmin, xmax, nx) |> collect

    ys = phi.(xs)

    datas = generatepriordata(ndata, rho_std)

    zs = perturb.(repmat(ys, zmult), rho_std)

    xs, ys, datas, zs
end



end
