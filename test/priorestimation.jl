# Unit test

wc = WeightedChain(rand(100,3), rand(100,5), rand(100))

emiteration!(wc)
euler_A!(wc, 1)
euler_phih!(wc, 1)
