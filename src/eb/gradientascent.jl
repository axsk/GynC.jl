function gradientascent(f::Function, w0::Vector, n::Integer, h::Real, projection::Function=GynC.projectsimplex; autodiff::Bool=true)
    df = autodiff ? gradify(f, w0) : f
    iter(w) = movefromboundary(w, projection(w + h * df(w)))
    collect(take(iterate(iter, w0), n))
end

function gradify(f, x)
    out = GradientResult(x)
    function df(x)
        ForwardDiff.gradient!(out, f, x)
        #ForwardDiff.value(out)
        ForwardDiff.gradient(out)
    end
end

" given two points on the simplex `wold` and `wnew`,
check whether `wnew` lies on the boundary and if so,
move it to linear combination of `wnew` and `wold` "
function movefromboundary(wold, wnew, relstep=1/2)
    if any(wnew .== 0)
        (1-relstep)*wold + relstep*wnew
    else
        wnew
    end
end
