Lausanne(id::Int) = Subject(lausannedata(id), (:lausanne, id))

function lausannedata(caseid::Int)
  dir = joinpath(dirname(@__FILE__), "lausaunne")
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
