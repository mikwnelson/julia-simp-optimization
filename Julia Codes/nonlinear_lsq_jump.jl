using JuMP, Ipopt

p = 500
n = 200

solver = with_optimizer(Ipopt.Optimizer);
ML = Model(solver)
@variable(ML, v[1:p] )

A = rand(n,p)
b = [ones(15); zeros(p-15)];
y = A*b+0.5*rand(n);

@variable(ML, z[1:size(y,1)])
@constraint(ML, z.==A*v-y)
@objective(ML, Min, sum(z.^2))

optimize!(ML)

println("got ", objective_value(ML), " at ", value(z[2]))