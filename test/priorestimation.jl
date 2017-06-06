# Unit test


info("testing WeightedChain")
wc = WeightedChain(rand(100,3), rand(100,5), rand(100))

info("testing emiteration!, euler_A!, euler_phih!")
emiteration!(wc)
euler_A!(wc, 1)
euler_phih!(wc, 1)


info("testing regularizers.jl")

N = 10
w = normalize!(rand(N), 1)

xs, ys, datas, zs = Federn.federexperiment(nx=N, ndata=N)
w = Federn.wbeta(xs, 110)


hz(w, ys, zs, 5)
logl(w, ys, datas, 5)

gradientascent(w->GynC.hz(w,ys,zs,0.1), w, 1, 0.1)

