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

  samples[idx,:], counts
end
