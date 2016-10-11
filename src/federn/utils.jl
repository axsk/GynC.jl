using ForwardDiff

function gradify(f, x)
    out = GradientResult(x)
    function df(x)
        ForwardDiff.gradient!(out, f, x)
        ForwardDiff.value(out), ForwardDiff.gradient(out)
    end
end

function gradientascent(f, w0, n, h, projection)
    df = gradify(f, w0)
    iter(w) = projection(w + h * df(w)[2])
    collect(take(iterate(iter, w0), n))
end
