using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, (m+1)*(n+1))

function av_temp(x::Vector, grad::Vector, Q, p, m, n)
    if length(grad) > 0
        dfav = d_f_av(x,p,m,n)
        grad[:] = dfav[:]
    end
    return f_av(x,Q,p,m,n)
end

function por(x::Vector, grad::Vector, m, n)
    if length(grad) > 0
        for i = 1:((m+1)*(n+1))
            grad[i] = 1/((m+1)*(n+1))
        end
    end
    return (1/((m+1)*(n+1)))*(ones(((m+1)*(n+1)),1)')*x
end

min_objective!(opt, (x,grad)->av_temp(x,grad,Q,p,m,n))

inequality_constraint!(opt, (x,grad)->por(x,grad,m,n))

opt.lower_bounds = 0
opt.upper_bounds = 1

opt.ftol_rel = 0.00001

η = convert(Vector, vec(Eta'))

(minf,minx,ret) = optimize!(opt, η)