speciesnames = open(readlines, "../data/speciesnames.txt")
parameternames = open(readlines, "../data/parameternames.txt")

""" load the (externally computed) maximal likelihood estimates """
function loadmles()
  parmat = matread("../data/parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  parms, y0
end

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function pfizerdata(person, path = "../data/pfizer_normal.txt")
  data = readtable(path, separator='\t')
  results = Vector()
  map(groupby(data, 6)) do subject
    p = fill(NaN, 4, 31)
    for measurement in eachrow(subject)
      # map days to 1-31
      day = (measurement[1]+30)%31+1
      for i = 1:4
        val = measurement[i+1]
        p[i,day] = isa(val, Number) ? val : NaN
      end
    end
    push!(results,p)
  end
  results[person]
end
