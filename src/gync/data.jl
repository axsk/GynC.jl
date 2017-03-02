function alldatas(; minmeas=20)
  datas = vcat(lausannedata(), pfizerdata())
  datas = filter(d->length(d) - sum(isnan(d)) >= minmeas, datas)
end

Lausanne(id::Int) = Patient(lausannedata()[id], "l$id")

function lausannedata()
  dir = joinpath(dirname(@__FILE__),"data", "lausaunne")
  daynames = map(Symbol, vcat(["_$i" for i = 16:-1:1], ["x$i" for i=0:15]))
  hormonefile = ["lh.csv", "fsh.csv", "oestr.csv", "prog.csv"]

  datas = [fill(NaN, 31, 4) for i=1:45]

  unitscaling = [1, 1, 1/3.671, 1]

  for h in 1:4 
    data = DataFrames.readtable(joinpath(dir,hormonefile[h]), separator=',')
    data = readdlm(joinpath(dir, hormonefile[h]), ',')

    for caseid in 1:45
      i = findfirst(data[:,1] .== caseid)
      i == 0 && continue
      for day in 1:31 # note we are losing 1 day of data here
        val = data[i, day+2]
        if isa(val, Number)
          datas[caseid][day, h] = val * unitscaling[h]
        end
      end
    end
  end

  datas
end

Pfizer(id::Int) = Patient(pfizerdata()[id], "p$id")

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function pfizerdata()
  file = joinpath(dirname(@__FILE__), "data", "pfizer_normal.txt")
  data = DataFrames.readtable(file, separator='\t')
  results = Vector()
  map(DataFrames.groupby(data, 6)) do subject
    p = fill(NaN, 31, 4)
    for measurement in DataFrames.eachrow(subject)
      # map days to 1-31
      day = (measurement[1]+30)%31+1
      for i = 1:4
        val = measurement[i+1]
        p[day, i] = isa(val, Number) ? val : NaN
      end
    end
    push!(results,p)
  end
  results
end
