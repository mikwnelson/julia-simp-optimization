using NLopt
include("SIMP-Optimization.jl")

opt = Opt(:LD_MMA, (m+1)*(n+1))

function av_temp(x::Vector, grad::Vector)
    if length(grad) > 0
        grad = d_f_av(x,p,m,n)
        return f_av(x,Q,p,m,n)
    end
end

#function myconstraint(x::Vector, grad::Vector, a, b)
#    if length(grad) > 0
#        grad[1] = 3a * (a*x[1] + b)^2
##        grad[2] = -1
 #   end
 #   (a*x[1] + b)^3 - x[2]
#end

opt.min_objective = av_temp
opt.lower_bounds = 0
opt.upper_bounds = 1

η = vec(Eta')

(minf,minx,ret) = optimize(opt, η)