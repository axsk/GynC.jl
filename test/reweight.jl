# Unit test

w = rand(3)
w = w / sum(w)
wc = SimpleWeightedChain(w, rand(3,2))

gradient_simplex!(wc,1)
error = abs(sum(wc.weights) - 1)
error < 1e-6 || error("gradient_simplex result has norm ", abs(sum(wc.weights)))
