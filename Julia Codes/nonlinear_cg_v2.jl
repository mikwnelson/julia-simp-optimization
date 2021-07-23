using LinearAlgebra

i = 0
k = 0

#Function to Optimize
f(x)=(x[2])^3-x[2]+(x[1])^2-3x[1]

#Gradient of Function
df(x)=[2x[1]-3,3x[2]^2-1]

#Hessian of Function
hf(x)=[2 0; 0 6x[2]]

#Initial Point
x = [0,0]

n = size(x)[1]

r = -df(x)

d = r

delta_new = (r')*r

delta_0 = delta_new

#Choose Max Iterations
i_max = 100

#Choose Max Newton-Raphson Iterations
j_max = 10

#Set CG Error Tolerance
epsilon_CG = 0.5

#Set Newton-Raphson Error Tolerance
epsilon_NR = 0.5

while (i < i_max) && (delta_new > (((epsilon_CG)^2)*(delta_0)))
    global j = 0
    global delta_d = (d')*d
    while true
        global alpha = -((df(x))'*d)/((d')*hf(x)*d)
        global x = x + alpha*d
        global j = j + 1
        (j < j_max) && ((alpha)^2*(delta_d) > (epsilon_NR)) || break
    end
    global r = -df(x)
    global delta_old = delta_new
    global delta_new = (r')*r
    global beta = (delta_new)/(delta_old)
    global d = r + beta*d
    global k = k + 1
    if (k == n) || (((r')*d) <= 0)
        global d = r
        global k = 0
    end
    global i = i + 1
end

display(x)
