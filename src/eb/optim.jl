using NLopt


OPT_LOWERXB = 1e-12
OPT_XTOLREL = 1e-5
OPT_XTOLABS = 0
OPT_FTOLREL = 1e-5
OPT_CTOLABS = 1e-5
OPT_DEBUG   = false
OPT_MAXEVAL = 0

" optimize the models objective using dmple_obj for derivatives, with simplex inequality constraints via augmented lagrangian"
function optimauglag(m, reg, w0; optimizer=:LD_MMA, maxeval=20)

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        OPT_DEBUG && @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), -f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) # - because of minimization
        else
            OPT_DEBUG && print("grad length $(length(g))")
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
    

    lower_bounds!(opt, OPT_LOWERXB)
    upper_bounds!(opt, 1)

    equality_constraint!(opt, eqconst, OPT_CTOLABS)


    min_objective!(opt, myf)
    #min_objective!(opt2, myf)

    #maxeval!(opt, 100)
    #maxeval!(opt2, 10)

    #ftol_abs!(opt2, 1)
    #ftol_abs!(opt, 1)
    ftol_rel!(opt, OPT_FTOLREL)
    ftol_rel!(opt2, OPT_FTOLREL)

    maxeval!(opt2, maxeval)
    xtol_rel!(opt, OPT_XTOLREL)
    xtol_rel!(opt2, OPT_XTOLREL)
    xtol_abs!(opt, OPT_XTOLABS)
    xtol_abs!(opt2, OPT_XTOLABS)

    local_optimizer!(opt, opt2)

    minf, minx, ret = optimize(opt, w0)

    if minx == w0
        warn("Optimizer returned initial proposal")
    end

    minx
end


" optimize the models objective using dmple_obj for derivatives, with simplex inequality constraints as solver constraints"
function optimineq(m, reg, w0; optimizer=:LD_MMA)

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        OPT_DEBUG && @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), -f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) 
        end
        -f(x)
    end

    function myineq(x,g)
        if length(g) == n
            g[:] = 1
        end
        sum(x) - 1. - OPT_CTOLABS/2
    end

    function myineq2(x,g)
        if length(g) == n
            g[:] = -1
        end
        1. - sum(x) - OPT_CTOLABS/2 
    end

    opt = Opt(optimizer,  n)

    lower_bounds!(opt, OPT_LOWERXB)
    upper_bounds!(opt, 1)

    inequality_constraint!(opt, myineq, OPT_CTOLABS/2)
    inequality_constraint!(opt, myineq2, OPT_CTOLABS/2)

    min_objective!(opt, myf)

    xtol_rel!(opt, OPT_XTOLREL)
    xtol_abs!(opt, OPT_XTOLABS)

    ftol_rel!(opt, OPT_FTOLREL)
    maxeval!(opt, OPT_MAXEVAL)


    minf, minx, ret = optimize(opt, w0)

    if minx == w0
        warn("Optimizer returned initial proposal")
    end

    minx
end
