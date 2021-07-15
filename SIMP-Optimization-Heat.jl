using NLopt, SparseArrays, LinearAlgebra, LaTeXStrings, Plots
pyplot()

##########################
## Fixed Variable Input ##
##########################

p = 1.0

m = 30

n = 40

k₀ = 1.0

k₊ = 100.0

xlen = 0.1

ylen = 0.1

ε₀ = 1e-3 # Outer loop error tolerance

εᵢ = 1e-4 # Inner Loop error tolerance

##########################
## Compute size of each ##
##   control volume     ##
##########################

Δx = xlen / n
Δy = ylen / m

###########################################
## Create Optimization Problem Structure ##
## Using MMA with dimentions (m+1)*(n+1) ##
###########################################

opt = Opt(:LD_MMA, (m + 1) * (n + 1))

#########################
## Average Temperature ##
## Objective Function  ##
#########################

function av_temp(
    η::Vector,
    grad::Vector,
    p,
    m,
    n,
    xlen = 0.1,
    ylen = 0.1,
    k₀ = 1.0,
    k₊ = 100.0,
)

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

    # Define Conductivity Penalization Function for design parameters eta
    k = k₀ .+ (k₊ - k₀) .* η .^ p

    # Control Volumes are designated based on matrix-type coordinates, so that volume [i,j] is the control volume in the i-th row and j-th column from the upper left.

    # Compute conductivites of temperature control volume boundaries
    # k_W[i,j] = conductivity of "West" boundary of [i,j] control volume

    k_W = 0.5 * (k[1:end-1, :] + k[2:end, :])

    # k_N[i,j] = conductivity of "North" boundary of [i,j] control volume

    k_N = 0.5 * (k[:, 1:end-1] + k[:, 2:end])

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
        K[cord2num(i, j, m), cord2num(i, j + 1, m)] = -k_W[i, j+1] * (Δy / Δx)
        K[cord2num(i, j + 1, m), cord2num(i, j, m)] = -k_W[i, j+1] * (Δy / Δx)
    end

    for i = 1:(m-1), j = 1:n
        K[cord2num(i, j, m), cord2num(i + 1, j, m)] = -k_N[i+1, j] * (Δx / Δy)
        K[cord2num(i + 1, j, m), cord2num(i, j, m)] = -k_N[i+1, j] * (Δx / Δy)
    end

    # Diagonal elements of K balance out column sums
    for j = 1:(m*n)
        K[j, j] = -sum(K[:, j])
    end

    ######################
    ## Add in effect of ##
    ##    Heat Sink     ##
    ######################

    # Add heat sink in middle of left side of material by adding conductivity to diagonal element of K in corresponding row
    if iseven(m)
        # Nearest half integer
        hm = m ÷ 2
        K[cord2num(hm, 1, m), cord2num(hm, 1, m)] += k_W[hm, 1] * (Δy / Δx)
        K[cord2num(hm + 1, 1, m), cord2num(hm + 1, 1, m)] += k_W[hm+1, 1] * (Δy / Δx)
    else
        hm = m ÷ 2 + 1
        K[cord2num(hm, 1, m), cord2num(hm, 1, m)] += k_W[hm, 1] * (Δy / Δx)
    end

    #######################
    ## Assemble Q Matrix ##
    #######################

    # Input vector of Heat-Generation rates
    Q = ones(m, n)

    ######################
    ## Compute T Vector ##
    ######################

    # Solve KT = Q
    T = K \ vec(Q)

    ###########################
    ## Gradient Computations ##
    ###########################

    if length(grad) > 0

        grad = reshape(grad, m + 1, n + 1)

        ############################
        ##    Compute λ vector    ##
        ## (Dual Vector for f_av) ##
        ############################

        λ = K \ (-ones((m * n), 1) * (1 / (m * n)))

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

            ######################
            ## Add in effect of ##
            ##    Heat Sink     ##
            ######################

            if iseven(m)
                hm = m ÷ 2
                if j == 1 && (hm ≤ i ≤ hm + 1)
                    dK[cord2num(hm, 1, m), cord2num(hm, 1, m)] += 0.5 * (Δy / Δx)
                end
                if j == 1 && (hm + 1 ≤ i ≤ hm + 2)
                    dK[cord2num(hm + 1, 1, m), cord2num(hm + 1, 1, m)] += 0.5 * (Δy / Δx)
                end
            else
                hm = m ÷ 2 + 1
                if j == 1 && (hm ≤ i ≤ hm + 1)
                    dK[cord2num(hm, 1, m), cord2num(hm, 1, m)] += 0.5 * (Δy / Δx)
                end
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
    end

    ##########################
    ## Compute f(T) = T_avg ##
    ##########################

    f_avg = sum(T) / (m * n)

    ########################
    ## Debugging Messages ##
    ########################

    #println("fav = $f_avg\n", "grad = $grad\n") #Output for Debugging Purposes

    return f_avg
end

##################################
## Porosity Constraint Function ##
##################################

function por(x::Vector, grad::Vector, m, n)
    if length(grad) > 0
        grad .= 1.0
    end
    con = sum(x) - 0.1 * (m + 1) * (n + 1)
    #println("con = $con\n", "grad = $grad\n") #Output for Debugging Purposes
    return con
end

###################################
## Add Objective and Constraints ## 
##       to opt structure        ##
###################################

min_objective!(opt, (x, g) -> av_temp(x, g, p, m, n))

inequality_constraint!(opt, (x, g) -> por(x, g, m, n), 1e-8)

opt.lower_bounds = 0
opt.upper_bounds = 1

opt.xtol_rel = εᵢ

##################
## Testing Code ##
##################

# η = rand( (m+1)*(n+1) ) # 0.1 .*ones((m+1)*(n+1))
# dη = 1e-3 * randn(size(η))
# grad = similar(η)

# av_temp(η, grad, p,m,n)
# @show norm(av_temp(η+dη, [], p,m,n) - av_temp(η, [], p,m,n) - dot(grad, dη))

# por(η, grad, m,n)
# @show norm(por(η+dη, [], m,n) - por(η, [], m,n) - dot(grad, dη))

# nothing

η = 0.05 .* ones((m + 1) * (n + 1))

#= 
counter = 1

while counter <= 3
    (minf, minx, ret) = optimize!(opt, η)
    numevals = opt.numevals # the number of function evaluations
    println("$p: $minf for $numevals iterations (returned $ret)")
    global p += 0.1
    global counter += 1
end =#

f_0 = 10 * av_temp(η, [], p, m, n)

while true
    (minf, minx, ret) = optimize!(opt, η)
    numevals = opt.numevals # the number of function evaluations
    println("$p: $minf for $numevals iterations (returned $ret)")
    global p += 0.05
    err = norm(minf - f_0)
    global f_0 = minf
    err <= ε₀ && break
end

# numevals = opt.numevals # the number of function evaluations
# println("got $minf at $minx after $numevals iterations (returned $ret)")

η = reshape(η, m + 1, n + 1)
# F = imshow(η, extent = (0.0, 0.1, 0.0, 0.1))

η_map = heatmap(
    0:Δx:xlen,
    0:Δy:ylen,
    η,
    yflip = true,
    xmirror = true,
    aspect_ratio = :equal,
    colorbar_title = "η",
    title = "η for each Design Volume",
)
