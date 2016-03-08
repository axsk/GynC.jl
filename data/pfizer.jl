Pfizer(id::Int)   = Subject(pfizerdata(), (:pfizer, id))

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function pfizerdata()
  data = readtable(joinpath(datadir,"pfizer_normal.txt"), separator='\t')
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
  results
end
