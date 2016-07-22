" compute the floating mean of Vector `v` with over `window` samples"
function floating_mean(v::Vector, window::Integer)
  window = window - 1
  map(i->mean(v[i:i+window]), 1:length(v)-window)
end

" rate of change, averaged over `window` samples " 
floating_acceptance(v::Vector, window) = 
  floating_mean(map(!=, v[1:end-1], v[2:end]), window)

floating_acceptance(c::Mamba.AbstractChains, window) = 
  floating_acceptance(c.value[:,1,1]|>vec, window)

" distance between samples, averaged over `window` samples " 
floating_distance(v::Vector, window) = 
  floating_mean(abs(map(-, v[1:end-1], v[2:end])), window)
