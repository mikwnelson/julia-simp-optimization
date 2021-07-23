using SparseArrays, LinearAlgebra

function Kay(Eta, k_0, k_p, p, m, n, xlen, ylen)

    #Compute size of each control volume
    #dx = "delta x"
    dx = xlen / n
    #dy = "delta y"
    dy = ylen / m

    #Define Conductivity Penalization Function for design parameters eta
    k = k_0 .+ (k_p - k_0) .* Eta .^ p

    #Control Volumes are designated based on matrix-type coordinates, so that volume [i,j] is the control volume in the i-th row and j-th column from the upper left.

    #Compute conductivites of temperature control volume boundaries
    #k_W[i,j] = conductivity of "West" boundary of [i,j] control volume

    k_W = 0.5 * (k[1:end-1, :] + k[2:end, :])

    #k_N[i,j] = conductivity of "North" boundary of [i,j] control volume

    k_N = 0.5 * (k[:, 1:end-1] + k[:, 2:end])

    #Initialize K matrix
    K = spzeros((m * n), (m * n))

    #Number control volumes based on node coordinates, going row-by-row, for m rows and n columns
    function cord2num(i, j, n)
        cv_num = j + (i - 1) * n
        return cv_num
    end

    #Construct K matrix
    #K[x,y] tells the heat flux from temperature volume number x to volume number y
    for i = 1:m, j = 1:(n-1)
        K[cord2num(i, j, n), cord2num(i, j + 1, n)] = -k_W[i, j+1] * (dy / dx)
        K[cord2num(i, j + 1, n), cord2num(i, j, n)] = -k_W[i, j+1] * (dy / dx)
    end

    for i = 1:(m-1), j = 1:n
        K[cord2num(i, j, n), cord2num(i + 1, j, n)] = -k_N[i+1, j] * (dx / dy)
        K[cord2num(i + 1, j, n), cord2num(i, j, n)] = -k_N[i+1, j] * (dx / dy)
    end

    #Diagonal elements of K balance out row sums
    for i = 1:(m*n)
        K[i, i] = -sum(K[i, :])
    end

    #Add heat sink in middle of left side of material by adding conductivity to diagonal element of K in corresponding row
    if iseven(m)
        #Nearest half integer
        hm = m ÷ 2
        K[cord2num(hm, 1, n), cord2num(hm, 1, n)] += k_W[hm, 1] * (dy / dx)
        K[cord2num(hm + 1, 1, n), cord2num(hm + 1, 1, n)] += k_W[hm+1, 1] * (dy / dx)
    else
        hm = m ÷ 2 + 1
        K[cord2num(hm, 1, n), cord2num(hm, 1, n)] += k_W[hm, 1] * (dy / dx)
    end
    #=
    #Add heat sink in middle of north side of material by adding conductivity to diagonal element of K in corresponding column
    if iseven(n)
        #Nearest half integer
        hm=n÷2
        K[cord2num(1,hm,n),cord2num(1,hm,n)]+=k_W[1,hm]*(dy/dx)
        K[cord2num(1,hm+1,n),cord2num(1,hm+11,n)]+=k_W[1,hm+1]*(dy/dx)
    else
        hm=n÷2+1
        K[cord2num(1,hm,n),cord2num(1,hm,n)]+=k_W[1,hm]*(dy/dx)
    end
    =#
    return K
end

#Compute relevant partial K matrix from eq (17) for design node (i,j)

#dK/dk
function partialK(i, j, m, n, xlen, ylen)

    #Compute size of each control volume
    #dx = "delta x"
    dx = xlen / n
    #dy = "delta y"
    dy = ylen / m

    DK = spzeros((m * n), (m * n))

    #Number control volumes based on node coordinates, going row-by-row, for m rows and n columns
    function cord2num(i, j, n)
        cv_num = j + (i - 1) * n
        return cv_num
    end

    if 2 <= j <= n
        for a = max(1, i - 1):min(i, m)
            DK[cord2num(a, j, n), cord2num(a, j - 1, n)] = -0.5 * (dy / dx)
            DK[cord2num(a, j - 1, n), cord2num(a, j, n)] = -0.5 * (dy / dx)
        end
    end
    if 2 <= i <= m
        for b = max(1, j - 1):min(j, n)
            DK[cord2num(i, b, n), cord2num(i - 1, b, n)] = -0.5 * (dx / dy)
            DK[cord2num(i - 1, b, n), cord2num(i, b, n)] = -0.5 * (dx / dy)
        end
    end
    for a = max(1, i - 1):min(i, m), b = max(1, j - 1):min(j, m)
        DK[cord2num(a, b, n), cord2num(a, b, n)] = -sum(DK[cord2num(a, b, n), :])
    end
    if iseven(m)
        hm = m ÷ 2
        if j == 1 && (hm ≤ i ≤ hm + 1)
            DK[cord2num(hm, 1, n), cord2num(hm, 1, n)] += 0.5 * (dy / dx)
        end
        if j == 1 && (hm + 1 ≤ i ≤ hm + 2)
            DK[cord2num(hm + 1, 1, n), cord2num(hm + 1, 1, n)] += 0.5 * (dy / dx)
        end
    else
        hm = m ÷ 2 + 1
        if j == 1 && (hm ≤ i ≤ hm + 1)
            DK[cord2num(hm, 1, n), cord2num(hm, 1, n)] += 0.5 * (dy / dx)
        end
    end
    return DK
end
