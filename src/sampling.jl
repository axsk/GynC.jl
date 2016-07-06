type Sampling
  samples::Matrix{Float64}
  config::Config
  variate::Mamba.AMMVariate
end

import Base.getindex

getindex(s::Sampling, i, j) = Sampling(s.samples[i,j], s.config, s.variate)

llh(s::Sampling) = pmap(x -> llh(s.config, x), [s.samples[k, :] |> vec for k in 1:size(s.aamples, 1)])

function solutions(s::Sampling, t=0:30)
  n = size(s.samples, 1)
  res = Array(Array, n)
  for k = 1:n
    res[k] = gync(s.samples[k,:] |> vec, t) 
  end
  res
end

function Base.show(io::IO, s::Sampling)
  print(io, "Sampling
  samples: $(size(s.samples))
  uniques: $(length(unique(s.samples[:,1])))
  config:  ", (s.config))
end

mean(s::Sampling) = s.variate.tune.Mv
sigma(s::Sampling) = s.variate.tune.SigmaLm

data(s::Sampling) = data(s.config)

function Sampling(c::Config)
  v = SamplerVariate(c)
  S = Matrix{Float64}(0, length(v[:]))

  Sampling(S, c, v)
end

sample(c::Config, iters::Int) = sample!(Sampling(c), iters)

function sample!(s::Sampling, iters::Int)
  thin = s.config.thin
  n    = round(Int, iters/thin, RoundDown)

  o, w = size(s.samples)
  S    = Array(Float64, o+n, w)
  S[1:o,:] = s.samples

  for i in 1:n
    for j in 1:thin
      Mamba.sample!(s.variate, adapt=s.config.adapt)
    end
    S[o+i,:] = unlist(s.variate[:])
  end

  s.samples = S
  s
end

#=

type Samples{T} <: AbstractArray{T, 2}
  samples :: Matrix{T}
  cumcounts  :: Vector{Int}
end

Samples{T}(n::Int, m::Int) = Samples{T}(zeros(T,n,m), zeros(Int,n))

Base.size(s::Samples) = size(s.samples)
Base.linearindexing(::Type{Samples}) = Base.LinearSlow()

index(i, cumcounts) = findfirst(c->c>=i, cumcounts)

function Base.getindex(s::Samples, i, j)
  s.samples[index(i, s.cumcounts),j]
end

function Base.setindex!(s::Samples, v, i, ::Colon)
  ii = index(i, s.cumcounts)
  if ii > 1 && s.samples[ii-1,:] == v
    info("saved space :)")
    s.cumcounts[ii-1] += 1
  else
    if s.cumcounts[ii] == 0
      s.cumcounts[ii] = 1
      s.samples[ii,:] = v
    else
      error("trying to overwrite value")
    end
  end
end

=#
