include("K_and_partialK.jl")

#Number of x-direction control volumes
n = 3

#Number of y-direction control volumes
m = 3

#Total number of temperature control volumes
N = m * n

#Input matrix of the design parameters (etas) at each design node
#Matrix values must be in [0,1]
Eta = 0.1 * ones(m + 1, n + 1)

#Input vector of Heat-Generation rates
Q = ones(m, n)

#Input thermal conductivites of low conductivity and high conductivity materials, k_0 and k_p, respectively
k_0 = 1
k_p = 100

#Initialize loop iterate
p = 2

#Set structure length dimensions
xlen = 0.1
ylen = 0.1

K1 = Kay(Eta, k_0, k_p, p, m, n, xlen, ylen)

t = 1e-4
h = ones(m + 1, n + 1)
Eta2 = Eta .+ t .* h

K2 = Kay(Eta2, k_0, k_p, p, m, n, xlen, ylen)

D = (K2 .- K1) ./ t


dk = (p * (k_p - k_0)) .* Eta .^ (p - 1) .* h
S = sum([partialK(i, j, m, n, xlen, ylen) .* dk[i, j] for i = 1:m+1, j = 1:n+1])

norm(D - S)
