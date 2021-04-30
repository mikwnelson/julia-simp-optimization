using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, N)

function av_temp(x::Vector, grad::Vector)
    if length(grad) > 0
        grad = d_f_av(x)
        return f_av(x)
    end
end

opt.min_objective = av_temp