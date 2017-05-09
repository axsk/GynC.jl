using ODE: ode45


@deprecate odeohnetreatment(k) odesol(k)

function odesol(k::Real, ts::AbstractVector = [1,1.7]; m = 70)
    #@show m
    f(t,z)   = [z[2], -k/m*z[1]]
    y0       = [-10, 0.]
    tspan    = vcat(0, ts)
    t,z      = ode45(f, y0, tspan, points=:specified)
    # return 2nd component at time 1 and 1.7
    [z[t][1] for t=2:length(z)]
end

function odemittreatment(k::Real; m = 70)

    s = 0.02
    g(t,c) = 400*exp(-(t-c).^2/s^2)

    c1 = 2.2;
    c2 = 2.97;
    c3 = 4.18;
    c4 = 4.95;
    c5 = 6.95;
    c6 = 7.75;
    treat(t) = g(t,c1) + g(t,c2) + g(t,c3) + g(t,c4) + g(t,c5) + g(t,c6);

    f(t,z) = [z[2], -k/m*z[1] + treat(t)]

    y0 = [-10,0.]
    tspan = linspace(0,10,400)

    t,y = ode45(f, y0, tspan, points=:specified)
end

function maxtreatamplitude(k)
    t,y = odemittreatment(k)
    ys = [y[1] for y in y]
    maxabs(ys)
end

r(k::Real) = maxtreatamplitude(k) > 13 ? 1 : 0

#=

ilyas alter code (bereits teilsangepasst)

function ODEmitTreatment3(kList)
    m     = 0.7;

    # F = @(t) 400 * ( exp(-(t-2.9).^2/0.02^2) + ...
    #     + exp(-(t-3.56).^2/0.02^2) + exp(-(t-4.24).^2/0.02^2) )...
    #     + 400 * (exp(-(t-5).^2/0.02^2) + exp(-(t-4.1).^2/0.02^2)...
    #     + exp(-(t-6.9).^2/0.02^2) + 2*exp(-(t-2.25).^2/0.02^2));

    s = 0.02;
    g(t,c) = 400*exp(-(t-c).^2/s^2);

    c1 = 2.2;
    c2 = 2.97;
    c3 = 4.18;
    c4 = 4.95;
    c5 = 6.95;
    c6 = 7.75;
    F(t) = g(t,c1) + g(t,c2) + g(t,c3) + g(t,c4) + g(t,c5) + g(t,c6);

    #=
    tt = linspace(0,10,20000)';
    figure('Position',[200,200,800,280])
    delt_x = 0.01;
    delt_y = 0.03;
    #Teil 1
    axes('Units', 'normalized', 'Position',[0+5*delt_x, 3*delt_y, 1-7*delt_x, 1-5*delt_y]); 
    plot(tt,F(tt))
    axis([0,10,0,500])
    l = legend('$F(t)$');
    set(l,'Interpreter','latex','Location','NorthEast','Fontsize',20)
    =#

    Test = zeros(kList)

    for ind = 1:length(kList)
        k = kList[ind];
        rhs(t,z) = [z[2], -k/m*z[1] + F(t)]
        y0 = [-10., 0]
        tspan = linspace(0,10.,400)
        T,A = ode45(rhs, y0, tspan)
        #Test[ind] = max(abs(A(1:400,1)));
        Test[ind] = maximum(abs(map(z->z[1], A)))
    end

    # figure
    # hold on
    # plot(kList,Test/20)
    # plot(kList,Test>=13)
    # axis([0 110 -0.2 1.2])

    # toc

    Test
end

=# 
