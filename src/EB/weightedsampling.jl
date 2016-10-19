type WeightedSampling{T}
  samples::Vector{T}
  weights::Vector{Float64}
end


function weightduplicates(samples::Matrix)
  idx    = Int[]
  counts = Int[]

  push!(idx, 1)
  push!(counts, 1)

  for i in 2:size(samples,1)
    if samples[i,:] == samples[i-1,:]
      counts[end] += 1
    else
      push!(idx, i)
      push!(counts, 1)
    end
  end

  samples[idx,:], counts / sum(counts)
end


function sample(s::WeightedSampling, n=1)
  cumdens = cumsum(s.weights)
  total   = cumdens[end]

  i = map(rand(n) * total) do target
    findfirst(x->x>=target, cumdens)
  end
  s.samples[i]
end
