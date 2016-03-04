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

type Subject
  dataset::Symbol
  id::Int
end

function data(s::Subject)
  if s.dataset == :pfizer
    pfizerdata()[s.id]
  elseif s.dataset == :lausanne
    lausannedata(s.id)
  else
    error("dataset not recognized")
  end
end

filename(s::Subject) = "$(s.dataset)$(s.id)"

function lausannedata(caseid::Int)
  dir = joinpath(datadir,"lausaunne")
  daynames = map(symbol, vcat(["_$i" for i = 16:-1:1], ["x$i" for i=0:15]))
  hormonefile = ["lh.csv", "fsh.csv", "oestr.csv", "prog.csv"]

  p = fill(NaN, 4, 31)

  for h in 1:4 
    data = readtable(joinpath(dir,hormonefile[h]), separator=',')
    try
      y = findfirst(data[:Cas].=="$caseid")
      for day in 1:32
        val = data[y,daynames[day]]
        day = (day-1)%31+1 # limex expects only 31 days 
        if isa(val, Number) 
          p[h,day] = val
        end
      end
    end
  end
  p
end
  

