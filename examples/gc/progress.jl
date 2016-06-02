using ProgressMeter

global p = Progress(0, 0.1)

function startprogress!(n)
  global p
  if p.counter == p.n
    p.counter = 0
    p.n = n
  end
end

function stepprogress!()
  global p
  next!(p)
end
