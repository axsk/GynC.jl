using ReverseDiff
using NLopt

" optimize f over the usimplex, using the orthogonal translation projection using ReverseDiff gradients "
function optimsimplex(f, x0; nmax=1000)
    evals = 0
    function fs(x) 
        evals += 1
        #f(x/sum(x))
        f(x-(sum(x)-1)/length(x))
    end
    dfs! = ReverseDiff.compile_gradient(fs, x0)

    opt = Opt(:LD_MMA, length(x0))

    lower_bounds!(opt, 0.)
    upper_bounds!(opt, 1.)

    min_objective!(opt, (x,g) -> (dfs!(g, x); fs(x)))

    maxeval!(opt, nmax)
    xtol_rel!(opt, 1e-6)
    xtol_abs!(opt, 1e-9)

    @time minf, minx, ret = optimize(opt, x0)
    @show evals
    @show minf
    @show ret

    @assert all(minx .>= 0) && all(minx .<= 1)
    minx / sum(minx)
end

" optimize the models objective using dmple_obj for derivatives, with simplex inequality constraints "
function optimmple(m, reg, w0; niter=100)
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)

    function myf(x,g)
        g[:] = -df(x) # - because of minimization
        -f(x)
    end

    opt = Opt(:LD_MMA, length(w0))
    lower_bounds!(opt, 0.)
    upper_bounds!(opt, 1.)

    # inequalities for sum(x) == 1 

    function myconst(x, grad)
        grad[:] = ones(grad)
        sum(x)-1
    end

    function myconst2(x,grad)
        grad[:] = -ones(grad)
        1-sum(x)
    end

    inequality_constraint!(opt, myconst, 1e-3)
    inequality_constraint!(opt, myconst2, 1e-3)

    min_objective!(opt, myf)

    maxeval!(opt, niter)
    #xtol_rel!(opt, 1e-6)
    #xtol_abs!(opt, 1e-9)

    minf, minx, ret = optimize(opt, w0)
    @show minf
    @show ret
    minx
end
