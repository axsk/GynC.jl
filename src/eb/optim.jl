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
function optimmple(m, reg, w0; optimizer=:LD_MMA, maxeval=20)

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), -f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) # - because of minimization
        else
            warn("grad length $(length(g))")
        end
        -f(x)
    end

    # inequalities for sum(x) == 1 

    function eqconst(x, grad)
        #grad[:] = ones(grad)
        fill!(grad, 1)
        (sum(x) - 1)
    end

    #=function myconst2(x,grad)
        grad[:] = -ones(grad)
        1-sum(x)
    end=#

    ### NLopt

    opt =  Opt(:LD_AUGLAG, n)
    opt2 = Opt(optimizer,  n)
    

    lower_bounds!(opt, 1e-12)
    upper_bounds!(opt, 1)

    equality_constraint!(opt, eqconst, 1e-4)


    min_objective!(opt, myf)
    #min_objective!(opt2, myf)

    #maxeval!(opt, 100)
    #maxeval!(opt2, 10)

    #ftol_abs!(opt, 1)
    #ftol_rel!(opt, 1e-6)
    #ftol_rel!(opt, 1e-6)
    #ftol_abs!(opt2, 1)

    maxeval!(opt2, maxeval)
    xtol_rel!(opt, 1e-3)
    xtol_rel!(opt2, 1e-3)
    xtol_abs!(opt, 1e-3 / n)
    xtol_abs!(opt2, 1e-3 / n)

    local_optimizer!(opt, opt2)

    @show minf, minx, ret = optimize(opt, w0)

    @assert minx != w0

    minx
end

" optimize the models objective using dmple_obj for derivatives, with simplex inequality constraints "
function optimmple2(m, reg, w0, penaltyconst; optimizer=:LD_MMA)

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), -f(x), sum(x.<0) + sum(x.>1))

        dsum = sum(x) - 1

        if length(g) == n
            g[:] = -df(x) + 2*dsum*x * penaltyconst # - because of minimization
        end

        -f(x) + dsum^2 * penaltyconst
    end

    opt = Opt(optimizer,  n)
    

    lower_bounds!(opt, 1e-12)
    upper_bounds!(opt, 1)



    min_objective!(opt, myf)

    xtol_rel!(opt, 1e-3)
    xtol_abs!(opt, 1e-3 / n)


    @show minf, minx, ret = optimize(opt, w0)

    @assert minx != w0

    minx
end
function optimmple3(m, reg, w0; optimizer=:LD_MMA)

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), -f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) 
        end
        -f(x)
    end

    function myineq(x,g)
        dsum = sum(x) - 1
        if length(g) == n
            g[:] = 1
        end
        dsum
    end

    opt = Opt(optimizer,  n)

    lower_bounds!(opt, 1e-12)
    upper_bounds!(opt, 1)

    inequality_constraint!(opt, myineq)

    min_objective!(opt, myf)

    xtol_rel!(opt, 1e-3)
    xtol_abs!(opt, 1e-3 / n)


    @show minf, minx, ret = optimize(opt, w0)

    @assert minx != w0

    minx
end
