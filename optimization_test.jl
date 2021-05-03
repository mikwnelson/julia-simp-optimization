using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, (m+1)*(n+1))

function av_temp(x::Vector, grad::Vector, Q, p, m, n)
    if length(grad) > 0
        dfav = d_f_av(x,p,m,n)
        grad[:] = dfav[:]
    end
    fav = f_av(x,Q,p,m,n)
    println("fav = $fav\n", "grad = $grad\n") #Output for Debugging Purposes
    return fav
end

function por(x::Vector, grad::Vector, m, n)
    if length(grad) > 0
        for i = 1:((m+1)*(n+1))
            grad[i] = 1/((m+1)*(n+1))
        end
    end
    con = (1/((m+1)*(n+1)))*(ones(((m+1)*(n+1)),1)')*x
    println("con = $con\n", "grad = $grad\n") #Output for Debugging Purposes
    return con
end

min_objective!(opt, (x,g) -> av_temp(x,g,Q,p,m,n))

inequality_constraint!(opt, (x,g) -> por(x,g,m,n), 1e-8)

opt.lower_bounds = 0
opt.upper_bounds = 1

opt.xtol_rel = 1e-4

η = convert(Vector, vec(Eta'))

(minf,minx,ret) = optimize!(opt, η)
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")