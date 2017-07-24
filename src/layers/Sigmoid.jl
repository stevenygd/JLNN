type Sigmoid <: Nonlinearity
    has_init    :: Bool
    x           :: Array{Float64}
    y           :: Array{Float64}
    dldy        :: Array{Float64}
    dldx        :: Array{Float64}

    function Sigmoid()
        return new(false, Float64[], Float64[], Float64[], Float64[])
    end
end

function init(l::Sigmoid, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    # TODO: currently I only accept Single dimensional dropout
    if p == nothing
        # [l] is the first layer, batch_size used default network batch_size
        # and input_size should be single dimensional (i.e. vector)
        @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
    else
        out_size = getOutputSize(p)
    end
    l.x = Array{Float64}(out_size)
    l.y = Array{Float64}(out_size)
    l.dldx = Array{Float64}(out_size)
    l.dldy = Array{Float64}(out_size)

    l.has_init = true
end

function update(l::Sigmoid, input_size::Tuple)
    l.x = Array{Float64}(input_size)
    l.y = Array{Float64}(input_size)
    l.dldx = Array{Float64}(input_size)
    l.dldy = Array{Float64}(input_size)
end

function forward(l::Sigmoid, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(l.x) != size(X)
        update(l, size(X))
    end
    l.x  = X
    l.y = map(x -> 1/(1+ e^(-x)), X)
    return l.y
end

function backward(l::Sigmoid, DLDY::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    @assert size(l.x) == size(DLDY)
    l.dldy = DLDY
    l.dldx = l.y .* (1 - l.y) .* DLDY # d(sigmoid(x))/dx = sigmoid(x)(1 - sigmoid(x))
    return l.dldx
end
