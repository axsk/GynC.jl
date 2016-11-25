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
  zs = repmat(ys, zmult) + rand(measerr, length(ys)*zmult)
  datas = phi.(rand(prior, ndata)) + rand(measerr, ndata)
  LikelihoodModel(xs, ys, zs, datas, measerr)
end

function smoothdata(m::LikelihoodModel, dmult::Int,
                    sigmak = KernelDensity.default_bandwidth(m.datas))
  # resample data
  sdatas = repmat(m.datas, dmult) + rand(Normal(0,sigmak), length(m.datas) * dmult);

  # adjust model sigma
  local smeaserr
  if isa(m.measerr, Normal)
    smeaserr = Normal(0, sqrt(m.measerr.σ^2 + sigmak^2))
  else
    warn("could not adjust the likelihoodmodel measurement error")
    smeaserr = m.measerr
  end

  # new model
  LikelihoodModel(m.xs, m.ys, m.zs, sdatas, smeaserr, m.zsampledistr)
end

function smoothdata(m::LikelihoodModel, dmult::Int, kernel::Distribution)
  #sdatas = repmat(m.datas, dmult) + rand(kernel, length(m.datas) * dmult)
  sdatas = map(d->d+rand(kernel), repmat(m.datas, dmult))

  warn("did not adjust the likelihoodmodel meas error")

  LikelihoodModel(m.xs, m.ys, m.zs, sdatas, m.measerr, m.zsampledistr)
end
