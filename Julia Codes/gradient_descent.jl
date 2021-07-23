using LinearAlgebra

#Function to Optimize
f(x)=(x[2])^3-x[2]+(x[1])^2-3x[1]

#Gradient of Function
df(x)=[2x[1]-3,3x[2]^2-1]

#Initial Point
x=[0,0]

#Line Search Algorithm
function ln_srch(d_dir,x,f,fx,dfx;alpha=0.25,beta=0.5)
    t = 1
    x1 = x+t*d_dir
    y1 = f(x1)
    y2 = fx+alpha*t*(dfx)'*d_dir
    while y1 > y2
        t = beta*t
        #println("t=",t, ", y1=", y1, ", y2=", y2)
        x1 = x+t*d_dir
        y1 = f(x1)
        y2 = fx+alpha*t*(dfx)'*d_dir
    end
    return t
end

#Gradient Descent Algorithm
function grad_d(f,df,x)
    d_dir = -df(x)
    t = ln_srch(d_dir,x,f,f(x),df(x))
    x = x + t*d_dir
    return x
end

#Compute Minimum for Defined Tolerance
while norm(df(x))>0.00001
    global x = grad_d(f,df,x)
end

display(x)
