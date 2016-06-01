# id: 1-13
# pfizer normal data

Pfizer(id::Int)   = Subject(pfizerdata()[id], (:pfizer, id))

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function pfizerdata()
  file = joinpath(dirname(@__FILE__), "pfizer_normal.txt")
  data = DataFrames.readtable(file, separator='\t')
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
