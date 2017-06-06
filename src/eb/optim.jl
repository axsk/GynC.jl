using NLopt
using Parameters

@with_kw type OptConfig
    LOWERXB = 1e-12
    XTOLREL = 1e-5
    XTOLABS = 0
    FTOLREL = 1e-5
    CTOLABS = 1e-5
    DEBUG   = false
    MAXEVAL = 0
    METHOD = :auglag
    OPTIMIZER = :LD_MMA
end

function optim(m, reg, w0; config=OptConfig())
    @unpack_OptConfig config

    # generate closures
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)
    
    function objective(x,g)
        # note the sign due to minimizing solver
        if length(g) == n
            g[:] = -df(x)
        end
        fx = f(x)
        DEBUG && @printf("f: f(x)=%f sum(x)=%f \n", fx, sum(x))
        -f(x)
    end

    local opt

    n = length(w0)
    opt = Opt(OPTIMIZER, n)

    min_objective!(opt, objective)
    lower_bounds!(opt, LOWERXB)
    upper_bounds!(opt, 1)

    xtol_rel!(opt, XTOLREL)
    xtol_abs!(opt, XTOLABS)
    ftol_rel!(opt, FTOLREL)

    maxeval!(opt, MAXEVAL)

    if METHOD == :ineq
        myineq(x, g)  = ((length(g) == n) && (g[:] = -1); 
                         sum(x) - 1. - CTOLABS/2)
        myineq2(x, g) = ((length(g) == n) && (g[:] = -1); 
                         1. - sum(x) - CTOLABS/2)

        inequality_constraint!(opt, myineq, CTOLABS/2)
        inequality_constraint!(opt, myineq2, CTOLABS/2)

    elseif METHOD == :auglag

        function eqconst(x, grad)
            fill!(grad, 1)
            sum(x) - 1
        end

        # store inner optimizer and overwrite with outer
        inneropt = opt
        opt  = Opt(:LD_AUGLAG, n)

        min_objective!(opt, objective)
        lower_bounds!(opt, LOWERXB)
        upper_bounds!(opt, 1)

        equality_constraint!(opt, eqconst, CTOLABS)


        xtol_rel!(opt, XTOLREL)
        xtol_abs!(opt, XTOLABS)
        ftol_rel!(opt, FTOLREL)

        local_optimizer!(opt, inneropt)
    end

    minf, minx, ret = optimize(opt, w0)

    if minx == w0
        warn("Optimizer returned initial proposal")
    end

    minx
end





" optimize the models objective using dmple_obj for derivatives, with simplex inequality constraints via augmented lagrangian"
function optimauglag(m, reg, w0; optimizer=:LD_MMA, config=OptConfig())
    @unpack_OptConfig config

    n = length(w0)
    

    ###  functions 
    f  = GynC.mple_obj(m,reg)
    df = GynC.dmple_obj(m,reg)


    function myf(x,g)
        DEBUG && @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) # - because of minimization
        else
            DEBUG && print("grad length $(length(g))")
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
    

    lower_bounds!(opt, LOWERXB)
    upper_bounds!(opt, 1)

    equality_constraint!(opt, eqconst, CTOLABS)


    min_objective!(opt, myf)
    #min_objective!(opt2, myf)

    #maxeval!(opt, 100)
    #maxeval!(opt2, 10)

    #ftol_abs!(opt2, 1)
    #ftol_abs!(opt, 1)
    ftol_rel!(opt, FTOLREL)
    ftol_rel!(opt2, FTOLREL)

    maxeval!(opt2, MAXEVAL)
    xtol_rel!(opt, XTOLREL)
    xtol_rel!(opt2, XTOLREL)
    xtol_abs!(opt, XTOLABS)
    xtol_abs!(opt2, XTOLABS)

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
        DEBUG && @printf("f: sum(x)=%f f(x)=%f outliers=%d \n", sum(x), f(x), sum(x.<0) + sum(x.>1))
        if length(g) == n
            g[:] = -df(x) 
        end
        -f(x)
    end

    function myineq(x,g)
        if length(g) == n
            g[:] = 1
        end
        sum(x) - 1. - CTOLABS/2
    end

    function myineq2(x,g)
        if length(g) == n
            g[:] = -1
        end
        1. - sum(x) - CTOLABS/2 
    end

    opt = Opt(optimizer,  n)

    lower_bounds!(opt, LOWERXB)
    upper_bounds!(opt, 1)

    inequality_constraint!(opt, myineq, CTOLABS/2)
    inequality_constraint!(opt, myineq2, CTOLABS/2)

    min_objective!(opt, myf)

    xtol_rel!(opt, XTOLREL)
    xtol_abs!(opt, XTOLABS)

    ftol_rel!(opt, FTOLREL)
    maxeval!(opt, MAXEVAL)


    minf, minx, ret = optimize(opt, w0)

    if minx == w0
        warn("Optimizer returned initial proposal")
    end

    minx
end
