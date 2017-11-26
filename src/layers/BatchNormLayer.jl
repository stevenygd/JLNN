# Define the Batch Normalization layers
include("LayerBase.jl")

type BatchNormLayer <: LearnableLayer
    has_init :: Bool

    # Input output place holders
    x        :: Array{Float64}       # (N, D) input
    y        :: Array{Float64}       # (N, D) output
    dldy     :: Array{Float64}       # (N, D) output gradient
    dldx     :: Array{Float64}       # (N, D) input gradient

    # Parameters
    gamma    :: Array{Float64}  # Scale parameter, shape (1,D)
    beta     :: Array{Float64}  # Offset parameter, shape (1,D)

    dgamma   :: Array{Float64}  # Scale gradient, shape(1,D)
    dbeta    :: Array{Float64}  # Offset gradient, shape(1,D)

    momentum :: Float64         # Update running mean/variance
    eps      :: Float64         # Numerical stability

    # Intermediates
    xhat     :: Array{Float64}  # (N,D)
    xmu      :: Array{Float64}  # (N,D)
    ivar     :: Array{Float64}  # (1,D)
    sqrtvar  :: Array{Float64}  # (1,D)
    var      :: Array{Float64}  # (1,D)


    function BatchNormLayer(momentum::Float64 = 0.9, eps::Float64 = 1e-5,
        N::Int = 1, D::Int = 1)
        @assert eps>0 && 0<momentum<1
        return new(false, zeros(N,D), zeros(N,D),
        zeros(N,D), zeros(N,D),
        ones(N,D), zeros(N,D),
        zeros(N,D), zeros(N,D),
        momentum, eps,
        Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

function init(l::BatchNormLayer, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    if p == nothing
        # Probably won't happen
        # TODO: what if this really happens?
    else
        out_size = getOutputSize(p)
        @assert length(out_size) == 2 # TODO: maybe a friendly error message?
    end
    N, D = out_size
    l.has_init = true
    gamma = ones(1,D)
    beta = zeros(1,D)
    l.gamma = gamma
    l.beta = beta

    l.x      =  Array{Float64}(out_size)
    l.y      =  Array{Float64}(out_size)
    l.dldx   =  Array{Float64}(out_size)
    l.dldy   =  Array{Float64}(out_size)
    l.dgamma =  Array{Float64}(1,D)
    l.dbeta  =  Array{Float64}(1,D)
end

function update(l::BatchNormLayer, input_size::Tuple;)
    l.x = Array{Float64}(input_size)
    l.y = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.dldy = Array{Float64}(input_size)
end

function forward(l::BatchNormLayer, X::Array{Float64}; deterministics=false)
    @assert length(size(X))==2
    if size(l.x) != size(X)
        update(l, size(X))
    end
    l.x = X
    N, D = size(X)
    l.y = Array{Float64}(N,D)

    # Calculate mean in each dimension and subtract
    mu = mean(X,1)
    xmu = Array{Float64}(N,D)
    broadcast!(-,xmu,X,mu)
    l.xmu = deterministics ? l.xmu : xmu
    # Calculate each dimension's variance
    sq = Array{Float64}(N,D)
    broadcast!(*,sq,xmu,xmu)
    var = mean(sq,1)
    l.var = deterministics ? l.var : var
    # Normalize data
    sqrtvar = Array{Float64}(1,D)
    broadcast!(^,sqrtvar,var,0.5)
    broadcast!(+,sqrtvar,sqrtvar,l.eps)
    l.sqrtvar = deterministics ? l.sqrtvar : sqrtvar
    ivar = Array{Float64}(1,D)
    broadcast!(/,ivar,1,sqrtvar)
    l.ivar = deterministics ? l.ivar : ivar
    xhat = Array{Float64}(N,D)
    broadcast!(*,xhat,xmu,l.ivar)
    l.xhat = deterministics ? l.xhat : xhat
    # Add scaling
    broadcast!(*,l.y,xhat,l.gamma)
    broadcast!(+,l.y,l.y,l.beta)

    return l.y
end


function backward(l::BatchNormLayer, dldy::Array{Float64}; kwargs...)
    @assert length(size(dldy))==2 && size(dldy)==size(l.x)
    N, D = size(dldy)
    # Intermediates
    l.dgamma = Array{Float64}(1,D)
    buffer = Array{Float64}(N,D)
    buffer_small = Array{Float64}(1,D)
    dxhat = Array{Float64}(N,D)
    dxmu1 = Array{Float64}(N,D)
    divar = Array{Float64}(1,D)
    dsqrtvar = Array{Float64}(1,D)
    dvar = Array{Float64}(1,D)
    dsq = Array{Float64}(N,D)
    dxmu2 = Array{Float64}(N,D)
    dx1 = Array{Float64}(N,D)
    dmu = Array{Float64}(1,D)
    dx2 = Array{Float64}(N,D)

    # Step 9
    l.dldy = dldy
    l.dbeta = sum(dldy,1)
    # Step 8
    broadcast!(*,buffer,dldy,l.xhat) #dgamma = sum(dldy * xhat, axis=1)
    l.dgamma = sum(buffer,1)
    broadcast!(*,dxhat,dldy,l.gamma)
    # Step 7
    broadcast!(*,dxmu1,dxhat,l.ivar)
    broadcast!(*,buffer,dxhat,l.xmu)
    divar = sum(buffer,1)
    # Step 6
    broadcast!(*,buffer_small,l.sqrtvar,l.sqrtvar) # buffer_small = sqrtvar^2
    broadcast!(/,buffer_small,-1,buffer_small)     # buffer_small = -1/(sqrtvar)^2
    broadcast!(*,dsqrtvar,buffer_small,divar)
    # Step 5
    broadcast!(^,buffer_small,(l.var+l.eps),-0.5)
    broadcast!(*,dvar,buffer_small,dsqrtvar)
    dvar = dvar * 0.5
    # Step 4
    broadcast!(*,dsq,1./N * ones(N,D),dvar)
    # Step 3
    broadcast!(*,dxmu2,2*dsq,l.xmu)
    # Step 2
    dx1 = dxmu1 + dxmu2
    dmu = -1 * sum(dxmu1+dxmu2,1)
    # Step 1
    broadcast!(*,dx2,1./N * ones(N,D),dmu)
    # Step 0
    l.dldx = dx1 + dx2
    return l.dldx
end

function getGradient(l::BatchNormLayer)
    return Array[l.dgamma, l.dbeta]
end

function setParam!(l::BatchNormLayer, theta)
    @assert size(l.gamma)==size(theta[1]) && size(l.beta)==size(theta[2])
    l.gamma = theta[1]
    l.beta = theta[2]
end

function getParam(l::BatchNormLayer)
    return Array[l.gamma, l.beta]
end

function getInputSize(l::BatchNormLayer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initialized. But input shape wanted.")
    end
    return size(l.x)
end

function getOutputSize(l::BatchNormLayer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initialized. But output shape wanted.")
    end
    return size(l.y)
end


# function test()
#     l = BatchNormLayer()
#     X = rand(3,5)
#     X[1,:] = [0.950802,0.162908,0.400773,0.895084,0.76084]
#     X[2,:] = [0.899969,0.917215,0.95278,0.632232,0.445346]
#     X[3,:] = [0.283375,0.592097,0.921092,0.868487,0.0497336]
#
#     out_size = (3,5)
#     l.has_init = true
#     l.gamma = ones(1,5)
#     l.beta = zeros(1,5)
#
#     l.x      =  Array{Float64}(out_size)
#     l.y      =  Array{Float64}(out_size)
#     l.dldx   =  Array{Float64}(out_size)
#     l.dldy   =  Array{Float64}(out_size)
#     l.dgamma =  Array{Float64}(1,5)
#     l.dbeta  =  Array{Float64}(1,5)
#
#     y = forward(l,X)
#     # print("\n\nX:\n")
#     # print(X)
#     # print("\n\ny:\n")
#     # print(y)
#
#     fake_dldy = ones(3,5)/2
#     dldx = backward(l,fake_dldy)
#     print("\ndldx:\n")
#     print(dldx)
#     print("\n\ndgamma:\n")
#     print(l.dgamma)
#     print("\n\ndbeta:\n")
#     print(l.dbeta)
#
# end

#test()
