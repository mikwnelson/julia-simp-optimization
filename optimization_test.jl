using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, (m+1)*(n+1))

function av_temp(x::Vector, grad::Vector)
    if length(grad) > 0
        dfav = d_f_av(x,p,m,n)
        for i = 1:((m+1)*(n+1))
            grad[i] = dfav[i]
        end
    end
        return f_av(x,Q,p,m,n)
end

function por(x::Vector, grad::Vector)
    if length(grad) > 0
        for i = 1:((m+1)*(n+1))
            grad[i] = 1/((m+1)*(n+1))
        end
    end
    return (1/((m+1)*(n+1)))*(ones(((m+1)*(n+1)),1)')*x
end

opt.min_objective = av_temp
opt.lower_bounds = 0
opt.upper_bounds = 1

opt.inequality_constraint = por

η = vec(Eta')

(minf,minx,ret) = optimize(opt, η)