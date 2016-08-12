using NLopt

count = 0

# marginal loglikelihood function
# L = sum{m} log(sum{k} L[k,m] * w[k])
function L(w,l)
	K, M = size(l)
	outer = 0.
	for m in 1:M
		inner = 0.
		for k in 1:K
			inner += w[k] * l[k,m]
		end
		outer += log(inner)
	end
	outer
end

@assert isa(L(rand(100),rand(100,6)), Real)

algorithms = [:LN_COBYLA, :LN_BOBYQA, :LN_NEWUOA_BOUND, :LN_PRAXIS, :LN_NELDERMEAD, :LN_SBPLX]

function optim(l::Matrix, algorithm=:LN_COBYLA, crit=10)
	K = size(l, 1)
	opt = Opt(algorithm, K)

	lower_bounds!(opt, zeros(K))
	upper_bounds!(opt, ones(K))

	inequality_constraint!(opt, (x,g) -> sum(x) - 1)

	#max_objective!(opt, (x,g) -> L(x,l) - abs2(sum(x) - 1) * 1e3)
	max_objective!(opt, (x,g) -> L(x,l))

	#ftol_abs!(opt, 1e-3)
	#xtol_rel!(opt, 1e-3)
	#xtol_abs!(opt, 1e-4)
	#maxtime!(opt, crit)
	maxeval!(opt, crit)

	optimize(opt, ones(K)/K)
end

using BenchmarkTools
l=rand(200,8)
@time optim(l)
