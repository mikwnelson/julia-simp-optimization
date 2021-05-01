using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, N)

function av_temp(x::Vector, grad::Vector)
    if length(grad) > 0
        grad = d_f_av(x,p,m,n)
        return f_av(x,Q,p,m,n)
    end
end

opt.min_objective = av_temp