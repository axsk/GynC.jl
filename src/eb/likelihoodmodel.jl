using KernelDensity

type LikelihoodModel
  xs::Vector
  ys::Vector
  zs::Vector
  datas::Vector
  measerr::Distribution
  zsampledistr::Distribution # used in hz (hy)
end

function LikelihoodModel(xs,ys,zs,datas,measerr)
  LikelihoodModel(xs,ys,zs,datas,measerr,measerr)
end


function em(m::LikelihoodModel, w0, niter)
  L = likelihoodmat(m.ys, m.datas, m.measerr)
  emiterations(w0, L, niter)
end

function mple(m::LikelihoodModel, w0, niter, reg, h)
  c1 = 1/100
  c2 = 1/1000
  ndata = length(m.datas)
  hauto = h/((1-reg)*(ndata/c1) + reg/c2)
  gradientascent(dmple_obj(m, reg), w0, niter, hauto, autodiff=false)
end


function mple_obj(m::LikelihoodModel, reg)
  if reg == 0
    w -> logl(m, w)
  elseif reg == 1
    w -> hz(m, w)
  else
    w -> reg*hz(m,w) + (1-reg) * logl(m,w)
  end
end

function dmple_obj(m::LikelihoodModel, reg)
  if reg == 0
    w -> dlogl(m, w)
  elseif reg == 1
    w -> dhz(m,w)
  else
    w -> reg*dhz(m, w) + (1-reg) * dlogl(m, w)
  end
end




hz(m::LikelihoodModel,  w) = hz(w,  m.ys, m.zs, m.zsampledistr)
dhz(m::LikelihoodModel, w) = dhz(w, m.ys, m.zs, m.zsampledistr)

logl(m::LikelihoodModel,  w) = logl(w,  m.datas, m.ys, m.measerr)
dlogl(m::LikelihoodModel, w) = dlogl(w, m.datas, m.ys, m.measerr) # correct order


function syntheticmodel(xs::Vector, phi::Function, prior::Distribution, ndata::Int, zmult::Int, measerr::Distribution)
  ys = phi.(xs)
  datas = phi.(rand(prior, ndata)) + rand(measerr, ndata)
  zs = repmat(ys, zmult) + rand(measerr, length(ys)*zmult)
  LikelihoodModel(xs, ys, zs, datas, measerr)
end


### smooth the data for computation of dsmle

function smoothedmodel(m, mult)
  smoothedmodel(m, mult, m.measerr)
end


function smoothedmodel(m, mult, _::Normal; sigma=KernelDensity.default_bandwidth(m.datas))
  kernel = Normal(0, sigma)

  datas = repmat(m.datas, mult)
  for i in eachindex(datas)
    datas[i] += rand(kernel)
  end
  
  measerr = Normal(0, sqrt(m.measerr.σ^2 + sigma^2))
  LikelihoodModel(m.xs, m.ys, m.zs, datas, measerr, m.zsampledistr)
end


" constructs the DSMLE model by inflating the data by factor mult.
adjusts the measurementerror for compensation"
function smoothedmodel{T}(m, mult, _::GynC.MatrixNormalCentered{T})
  sigmas = defaultdatabandwith(m)
  kernel = GynC.MatrixNormalCentered(sigmas)

  datas = repmat(m.datas, mult)
  for i in eachindex(datas)
    datas[i] += rand(kernel)
  end

  sigmas = sqrt.(m.measerr.sigmas.^2 + kernel.sigmas.^2)
  measerr = GynC.MatrixNormalCentered(sigmas)

  LikelihoodModel(m.xs, m.ys, m.zs, datas, measerr, m.zsampledistr)
end

function defaultdatabandwith(m::LikelihoodModel)
  datas = m.datas
  sigmas = similar(m.measerr.sigmas)
  for i in eachindex(sigmas)
    points = filter(x->!isnan(x), [d[i] for d in datas])
    sigmas[i] = KernelDensity.default_bandwidth(points)
  end
  sigmas
end
