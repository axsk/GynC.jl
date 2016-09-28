# Unit test

K = 100

wc = WeightedChain(rand(100,3), rand(100,5), rand(100))

emiteration!(wc)
euler_A!(wc, 1)
euler_phih!(wc, 1)

GynC.hzobj(rand(K,116), [rand(31,4) for i in 1:3])(rand(K))
