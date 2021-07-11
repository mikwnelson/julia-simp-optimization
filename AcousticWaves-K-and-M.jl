##########################
## Fixed Variable Input ##
##########################

m = 10

n = 10

A₁ = 1.0

A₂ = 100.0

B₁ = 1.0

B₂ = 50.0

μ_A = A₂ / A₁

μ_B = B₂ / B₁

xlen = 0.1

ylen = 0.1

η = 0.5 .* ones((m + 1) * (n + 1))

#######################
## Assemble K Matrix ##
#######################

##########################
## Compute size of each ##
##   control volume     ##
##########################

Δx = xlen / n
Δy = ylen / m

η = reshape(η, m + 1, n + 1)

# Define Interpolation Functions for design parameters eta for each material

A_η = (1 + η .* (μ_A - 1)) .* A₁

B_η = (1 + η .* (μ_B - 1)) .* B₁

# Control Volumes are designated based on matrix-type coordinates, so that volume [i,j] is the control volume in the i-th row and j-th column from the upper left.

# Compute Material Composition of control volume boundaries
# A_W[i,j] = conductivity of "West" boundary of [i,j] control volume

A_W = 0.5 * (A_η[1:end-1, :] + A_η[2:end, :])

# A_N[i,j] = conductivity of "North" boundary of [i,j] control volume

A_N = 0.5 * (A_η[:, 1:end-1] + A_η[:, 2:end])

# Initialize K matrix
K = spzeros((m * n), (m * n))

# Number control volumes based on node coordinates, going column-by-column, for m rows and n columns
function cord2num(i, j, m)
    cv_num = i + (j - 1) * m
    return cv_num
end

# Construct K matrix
# K[x,y] tells the heat flux from temperature volume number x to volume number y
for i = 1:m, j = 1:(n-1)
    K[cord2num(i, j, m), cord2num(i, j + 1, m)] = -A_W[i, j+1] * (Δy / Δx)
    K[cord2num(i, j + 1, m), cord2num(i, j, m)] = -A_W[i, j+1] * (Δy / Δx)
end

for i = 1:(m-1), j = 1:n
    K[cord2num(i, j, m), cord2num(i + 1, j, m)] = -A_N[i+1, j] * (Δx / Δy)
    K[cord2num(i + 1, j, m), cord2num(i, j, m)] = -A_N[i+1, j] * (Δx / Δy)
end

# Diagonal elements of K balance out column sums
for j = 1:(m*n)
    K[j, j] = -sum(K[:, j])
end

#######################
## Assemble M Matrix ##
#######################



#=
#########################
## Create ∂k/∂η Matrix ##
#########################

dk = (p * (k₊ - k₀)) .* η .^ (p - 1)

###########################
## Assemble ∂K/∂η Matrix ##
###########################

for i = 1:m+1, j = 1:n+1

    ###########################
    ## Assemble ∂K/∂k Matrix ##
    ##     for each (i,j)    ##
    ###########################

    dK = spzeros((m * n), (m * n))

    if 2 <= j <= n
        for a = max(1, i - 1):min(i, m)
            dK[cord2num(a, j, m), cord2num(a, j - 1, m)] = -0.5 * (Δy / Δx)
            dK[cord2num(a, j - 1, m), cord2num(a, j, m)] = -0.5 * (Δy / Δx)
        end
    end
    if 2 <= i <= m
        for b = max(1, j - 1):min(j, n)
            dK[cord2num(i, b, m), cord2num(i - 1, b, m)] = -0.5 * (Δx / Δy)
            dK[cord2num(i - 1, b, m), cord2num(i, b, m)] = -0.5 * (Δx / Δy)
        end
    end
    for a = max(1, i - 1):min(i, m), b = max(1, j - 1):min(j, n)
        dK[cord2num(a, b, m), cord2num(a, b, m)] = -sum(dK[cord2num(a, b, m), :])
    end

    ###########################
    ## Find Nonzero elements ##
    ##  of ∂K/∂η_{i,j} and   ##
    ## Assemble ∂f_av Matrix ##
    ###########################
    grad[i, j] = 0.0
    A, B, Va = findnz(dK)
    for k = 1:nnz(dK)
        a = A[k]
        b = B[k]
        v = Va[k]
        grad[i, j] += λ[a] * v * T[b]
    end

    #############################
    ## (∂K/∂k)*(∂k/∂η) = ∂K/∂η ##
    #############################

    grad[i, j] *= dk[i, j]
end
=#