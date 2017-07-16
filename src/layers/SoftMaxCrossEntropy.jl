# include("LayerBase.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    dldx   :: Array{Float64} # backprop
    x      :: Array{Float64} # input vector
    y      :: Array{Float64} # output of softmax
    exp    :: Array{Float64} # cache for exp(x)
    lsum   :: Array{Float64} # cache for sum(exp,2)
    loss   :: Array{Float64} # output of cross entropy loss
    pred   :: Array{Int64}   # output for prediction

    function SoftMaxCrossEntropyLoss()
        return new(Float64[], Float64[], Float64[], Float64[], Float64[], Int64[], 1:1)
    end
end

function init(l::SoftMaxCrossEntropyLoss, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
    # TODO: currently I only accept Single dimensional dropout
    if p == nothing
        # [l] is the first layer, batch_size used default network batch_size
        # and input_size should be single dimensional (i.e. vector)
        @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
        out_size = (config["batch_size"], config["input_size"][1])
    else
        out_size = getOutputSize(p)
    end
    N, D     = out_size
    l.dldx   = Array{Float64}(out_size)
    l.x      = Array{Float64}(out_size)
    l.y      = Array{Float64}(out_size)
    l.loss   = Array{Float64}(N)
    l.exp    = Array{Float64}(out_size)
    l.pred   = Array{Int64}(N)
    l.lsum   = zeros(size(l.x))
end

function update(l::SoftMaxCrossEntropyLoss, input_size::Tuple;)
    # We only allow to update the batch size
    @assert length(input_size) == 2
    @assert input_size[2] == size(l.x, 2)
    N, D = input_size[1], size(l.x, 2)
    l.dldx   = Array{Float64}(input_size)
    l.x      = Array{Float64}(input_size)
    l.y      = Array{Float64}(input_size)
    l.loss   = Array{Float64}(N)
    l.pred   = Array{Int64}(N)
    l.lsum = zeros(size(l.x))
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)
    @assert size(Y, 2) == size(l.x, 2)
    local N = size(Y, 1)
    if N != size(l.x, 1)
      update(l, size(Y))
    end
    l.x = Y
    l.exp = exp(l.x)
    l.lsum = sum(l.exp,2)
    l.y = l.exp ./ l.lsum

    l.loss = - sum(log(l.y) .* label,2)

    return l.loss, l.x
end

function backward(l::SoftMaxCrossEntropyLoss, label::Array{Float64, 2};kwargs...)

    l.dldx = l.y .* sum(label,2) - label

    return l.dldx
end
