using NLopt, SparseArrays, LinearAlgebra

##########################
## Fixed Variable Input ##
##########################

p = 1

m = 3

n = 3

###########################################
## Create Optimization Problem Structure ##
## Using MMA with dimentions (m+1)*(n+1) ##
###########################################

opt = Opt(:LD_MMA, (m+1)*(n+1))

#########################
## Average Temperature ##
## Objective Function  ##
#########################

function av_temp(η::Vector, grad::Vector, p, m, n)
    
    #######################
    ## Assemble K Matrix ##
    #######################

    #Compute size of each control volume
    #dx = "delta x"
    dx = xlen/n
    #dy = "delta y"
    dy = ylen/m

    Eta = reshape(η,m+1,n+1)

    #Define Conductivity Penalization Function for design parameters eta
    k = k_0 .+ (k_p - k_0).*Eta.^p

    #Control Volumes are designated based on matrix-type coordinates, so that volume [i,j] is the control volume in the i-th row and j-th column from the upper left.

    #Compute conductivites of temperature control volume boundaries
    #k_W[i,j] = conductivity of "West" boundary of [i,j] control volume

    k_W = 0.5*(k[1:end-1,:]+k[2:end,:])

    #k_N[i,j] = conductivity of "North" boundary of [i,j] control volume

    k_N = 0.5*(k[:,1:end-1]+k[:,2:end])

    #Initialize K matrix
    K = spzeros((m*n),(m*n))

    #Number control volumes based on node coordinates, going row-by-row, for m rows and n columns
    function cord2num(i,j,n)
        cv_num = j + (i-1)*n
        return cv_num
    end

    #Construct K matrix
    #K[x,y] tells the heat flux from temperature volume number x to volume number y
    for i=1:m, j=1:(n-1)
        K[cord2num(i,j,n),cord2num(i,j+1,n)]=-k_W[i,j+1]*(dy/dx)
        K[cord2num(i,j+1,n),cord2num(i,j,n)]=-k_W[i,j+1]*(dy/dx)
    end

    for i=1:(m-1), j=1:n
        K[cord2num(i,j,n),cord2num(i+1,j,n)]=-k_N[i+1,j]*(dx/dy)
        K[cord2num(i+1,j,n),cord2num(i,j,n)]=-k_N[i+1,j]*(dx/dy)
    end

    #Diagonal elements of K balance out row sums
    for i=1:(m*n)
        K[i,i]=-sum(K[i,:])
    end

    #Add heat sink in middle of left side of material by adding conductivity to diagonal element of K in corresponding row
    if iseven(m)
        #Nearest half integer
        hm=m÷2
        K[cord2num(hm,1,n),cord2num(hm,1,n)]+=k_W[hm,1]*(dy/dx)
        K[cord2num(hm+1,1,n),cord2num(hm+1,1,n)]+=k_W[hm+1,1]*(dy/dx)
    else
        hm=m÷2+1
        K[cord2num(hm,1,n),cord2num(hm,1,n)]+=k_W[hm,1]*(dy/dx)
    end

    #######################
    ## Assemble Q Matrix ##
    #######################

    #Input vector of Heat-Generation rates
    Q = ones(m,n)

    ######################
    ## Compute T Vector ##
    ######################

    #Solve KT = Q
    T = K\Q[:]

    ###########################
    ## Gradient Computations ##
    ###########################

    if length(grad) > 0

        d_f_av = zeros(m+1,n+1)

        ############################
        ##    Compute λ vector    ##
        ## (Dual Vector for f_av) ##
        ############################

        λ = K\(-ones((m*n),1)*(1/(m*n)))

        #########################
        ## Create ∂k/∂η Matrix ##
        #########################

        dk = (p * (k_p - k_0)) .* Eta.^(p-1)

        ###########################
        ## Assemble ∂K/∂η Matrix ##
        ###########################

        for i = 1:n+1, j = 1:m+1

            ###########################
            ## Assemble ∂K/∂k Matrix ##
            ##     for each (i,j)    ##
            ###########################

            DK = spzeros((m*n),(m*n))

            if 2 <= j <= n
                for a = max(1,i-1):min(i,m)
                    DK[cord2num(a,j,n),cord2num(a,j-1,n)]=-0.5*(dy/dx)
                    DK[cord2num(a,j-1,n),cord2num(a,j,n)]=-0.5*(dy/dx)
                end
            end
            if 2 <= i <= m
                for b = max(1,j-1):min(j,n)
                    DK[cord2num(i,b,n),cord2num(i-1,b,n)]=-0.5*(dx/dy)
                    DK[cord2num(i-1,b,n),cord2num(i,b,n)]=-0.5*(dx/dy)
                end
            end
            for a = max(1,i-1):min(i,m), b = max(1,j-1):min(j,m)
                DK[cord2num(a,b,n),cord2num(a,b,n)]=-sum(DK[cord2num(a,b,n),:])
            end
            if iseven(m)
                hm=m÷2
                if j==1 && (hm ≤ i ≤ hm + 1)
                    DK[cord2num(hm,1,n),cord2num(hm,1,n)]+=0.5*(dy/dx)
                end
                if j==1 && (hm +1 ≤ i ≤ hm + 2)
                    DK[cord2num(hm+1,1,n),cord2num(hm+1,1,n)]+=0.5*(dy/dx)
                end
            else
                hm=m÷2+1
                if j==1 && (hm ≤ i ≤ hm + 1)
                    DK[cord2num(hm,1,n),cord2num(hm,1,n)]+=0.5*(dy/dx)
                end
            end

            #############################
            ## (∂K/∂k)*(∂k/∂η) = ∂K/∂η ##
            #############################

            DK = DK.*dk[i,j]

            ###########################
            ## Find Nonzero elements ##
            ##  of ∂K/∂η_{i,j} and   ##
            ## Assemble ∂f_av Matrix ##
            ###########################

            A, B, Va = findnz(dK)
            for k = 1:nnz(dK)
                a = A[k]
                b = B[k]
                v = Va[k]
                d_f_av[i,j]+=λ[a]*v*T[b]
            end
        end

        ############################
        ## Vectorize ∂f_av Matrix ##
        ############################

        ∂f_av = convert(Vector, vec((d_f_av)'))

        #############################
        ## Input Elements of ∂f_av ##
        ##   into 'grad' vector    ##
        #############################

        grad[:] = dfav[:]

    end

    ##########################
    ## Compute f(T) = T_avg ##
    ##########################

    f_avg = (1/(m*n))*(ones((m*n),1))'*T

    ########################
    ## Debugging Messages ##
    ########################

    println("fav = $fav\n", "grad = $grad\n") #Output for Debugging Purposes

    return f_avg
end

##################################
## Porosity Constraint Function ##
##################################

function por(x::Vector, grad::Vector, m, n)
    if length(grad) > 0
        for i = 1:((m+1)*(n+1))
            grad[i] = 1/((m+1)*(n+1))
        end
    end
    con = (1/((m+1)*(n+1)))*(ones(((m+1)*(n+1)),1)')*x
    println("con = $con\n", "grad = $grad\n") #Output for Debugging Purposes
    return con
end


min_objective!(opt, (x,g) -> av_temp(x,g,p,m,n))

inequality_constraint!(opt, (x,g) -> por(x,g,m,n), 1e-8)

opt.lower_bounds = 0
opt.upper_bounds = 1

opt.xtol_rel = 1e-4

η = 0.1 .*ones((m+1)*(n+1))

(minf,minx,ret) = optimize!(opt, η)
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")