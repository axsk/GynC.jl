# Unit test
using FactCheck

facts("testing WeightedChain") do

  wc = WeightedChain(rand(100,3), rand(100,5), rand(100))
  emiteration!(wc)
  euler_A!(wc, 1)
  euler_phih!(wc, 1)
end


facts("testing regularizers.jl") do

  N = 10
  w = normalize!(rand(N), 1)

  xs, ys, datas, zs = Federn.federexperiment(nx=N, ndata=N)
  w = Federn.wbeta(xs, 110)


  @pending hz(w, ys, zs, 5)
  @pending logLw(w, ys, datas, 5)

  @pending gradientascent(w->GynC.Hz(w,ys,zs,0.1), w, 1, 0.1)
end


facts("this lead to problems (optimization didnt terminate, generates NaN)") do

  xs = linspace(1,50,100) |> collect
  ts = [0.3]
  phi(k) = GynC.Federn.odesol(k, ts)[1]

  measerr = Normal(0,0.1)
  ndata = 100
  augz = 20
  prior = GynC.Federn.prior

  srand(1)
  m = GynC.syntheticmodel(xs, phi, prior, ndata, augz, measerr)

  augd = 20
  stdd = KernelDensity.default_bandwidth(m.datas)
  ms = GynC.smoothedmodel(m, augd, Normal(0,stdd))

  w0 = ones(length(xs)) / length(xs);

  context("testing optim") do
    r = .5
    GynC.optim(m, .5, w0)
    GynC.optim(m, .5, w0, config=GynC.OptConfig(METHOD=:auglag))
  end

  context("testing mple") do
    GynC.mple(m, .5, 1, 10)
  end

end

exitstatus()
