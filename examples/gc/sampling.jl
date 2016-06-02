type Sampling
  samples::Array
  logprior::Vector
  logpost::Vector
  config::Config
  variate::Mamba.AMMVariate
  thin::Int
end

llh(s::Sampling) = pmap(x -> llh(s.config, x), [s.samples[k, :] |> vec for k in 1:size(s.samples, 1)])

function solutions(s::Sampling, t=0:30)
  n = size(s.samples, 1)
  res = Array(Array, n)
  for k = 1:n
    res[k] = gync(s.config, s.samples[k,:] |> vec, t) 
  end
  res
end
  

function Base.show(io::IO, s::Sampling)
  print(io, "Sampling
  samples: $(size(s.samples))
  uniques: $(length(unique(s.samples[:,1])))
  logpost: $(hist(s.logpost))
  config:  $(s.config)
  thin:    $(s.thin)")
end

mean(s::Sampling) = s.variate.tune.Mv
sigma(s::Sampling) = s.variate.tune.SigmaLm

function Sampling(c::Config, v::Mamba.SamplerVariate, t::Int)
  Sampling(Matrix{Float64}(0, length(v[:])), Float64[], Float64[], c, v, t)
end

function sample(c::Config, iters::Int; thin=1::Int)
  v = SamplerVariate(c)
  sample!(Sampling(c, v, thin), iters)
end

function sample!(s::Sampling, iters::Int)
  #startprogress!(iters)
  thin  = s.thin
  n     = round(Int, iters/thin, RoundDown)
  v     = s.variate
  x     = Array(Float64, n, length(v[:]))
  priors = Array(Float64, n)
  posts  = Array(Float64, n)

  post  = get(v.tune.logf) # use cached version

  for i in 1:n
    for j in 1:thin
      Mamba.sample!(v)
      #stepprogress!()
    end
    u = unlist(v[:])
    x[i,:] = u
    #priors[i] = prior(s.config, u)
    #posts[i]  = post(v[:])
  end

  Sampling(
    vcat(s.samples, x),
    vcat(s.logprior, priors),
    vcat(s.logpost, posts),
    s.config,
    v,
    s.thin)
end
