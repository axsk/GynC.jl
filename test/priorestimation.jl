# Unit test

K = 100

info("testing WeightedChain")
wc = WeightedChain(rand(100,3), rand(100,5), rand(100))

info("testing emiteration!, euler_A!, euler_phih!")
emiteration!(wc)
euler_A!(wc, 1)
euler_phih!(wc, 1)

info("testing hzobj")
hzobj(rand(K,116), [rand(31,4) for i in 1:3])(rand(K))

