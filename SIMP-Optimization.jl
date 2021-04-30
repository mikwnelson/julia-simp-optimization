using SparseArrays, LinearAlgebra, Plots, LaTeXStrings
include("K_and_partialK.jl")

##############
### Inputs ###
##############

#Number of x-direction control volumes
n = 100

#Number of y-direction control volumes
m = 100

#Total number of temperature control volumes
N = m*n

#Input matrix of the design parameters (etas) at each design node
#Matrix values must be in [0,1]
Eta = 0.1 * ones(m+1,n+1)
#Eta = rand(m+1,n+1)

#Input vector of Heat-Generation rates
Q = ones(m,n)

#Input thermal conductivites of low conductivity and high conductivity materials, k_0 and k_p, respectively
k_0 = 1
k_p = 100

#Initialize loop iterate
p = 2

#Set structure length dimensions
xlen = 0.1
ylen = 0.1

####################
### Computations ###
####################

#Compute size of each control volume
#dx = "delta x"
dx = xlen/n
#dy = "delta y"
dy = ylen/m

#Construct K and dK matrices
K = Kay(Eta, k_0, k_p, p, m, n, xlen, ylen)

#Compute Temperature Vector
T = K\Q[:]

#Reshape Temperature Vector as Grid
T_grid = reshape(T,n,m)'

#Heatmap Plot of Temperature
P = heatmap(0:dx:xlen,0:dy:ylen,T_grid,yflip=true,xmirror=true, title=L"T",colorbar_title=" ")

#Compute Average Temperature
f_av = (1/N)*(ones(N,1))'*(T)

#Compute Dual Vector for Average Temperature
lambda = K\(-ones(N,1)*(1/N))

#Computation of d_f_av
d_f_av = zeros(m+1,n+1)

#Derivative of penalization function
dk = (p * (k_p - k_0)) .* Eta.^(p-1)

for i = 1:n+1, j = 1:m+1
    dK = partialK(i,j,m,n,xlen,ylen).*dk[i,j]
    A, B, Va = findnz(dK)
    for k = 1:nnz(dK)
        a = A[k]
        b = B[k]
        v = Va[k]

        d_f_av[i,j]+=lambda[a]*v*T[b]
    end
end

#Heatmap of change in average temperature per design node
F = heatmap(0:dx:xlen,0:dy:ylen,d_f_av,yflip=true,xmirror=true,colorbar_title=" ",title=L"\textrm{d}f_{av}/\textrm{d}\eta_{ij}")

#Make d_f_av into a vector
d_f_av_vec = vec((d_f_av)')