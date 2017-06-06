# include("LayerBase.jl")

type SoftMaxCrossEntropyLoss <: LossCriteria
    dldx   :: Array{Float64}
    x      :: Array{Float64}
    y      :: Array{Float64}
    target :: Array{Float64}
    loss   :: Array{Float64}
    pred   :: Array{Int64}
    label  :: Array{Int64}
    iter   :: UnitRange{Int64}

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
    l.pred   = Array{Int64}(N)
    l.label  = Array{Int64}(N)
    l.iter   = 1:N
    l.target = zeros(size(l.x))
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
    l.label  = Array{Int64}(N)
    l.iter   = 1:N
    l.target = zeros(size(l.x))
end

function forward(l::SoftMaxCrossEntropyLoss, Y::Array{Float64,2}, label::Array{Int, 2}; kwargs...)
    # TODO this is a questionable invariant; [label] ranges from 0 to 9 instead of being one hot
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    @assert size(Y, 2) == size(l.x, 2)
    local N = size(Y, 1)
    if N != size(l.x, 1)
        update(l, size(Y))
    end
    broadcast!(-, l.x, Y, maximum(Y, 2))
    broadcast!(-, l.y, log(sum(exp(l.x),2)), l.x)
    for i = 1:N
        l.label[i] = label[i] + 1
    end
    # map!(x -> convert(Int64, x) + 1,    l.label, label)

    for i = 1:N
         l.loss[i] = l.y[i, l.label[i]]
    end
    # map!(i -> l.y[i, l.label[i]],       l.loss,  l.iter)

    for i = 1:N
        l.pred[i] = findmax(Y[i,:])[2] - 1
    end
    # map!(i -> findmax(Y[i,:])[2] - 1,   l.pred,  l.iter)

    return l.loss, l.pred
end

function backward(l::SoftMaxCrossEntropyLoss, label::Array{Int, 2};kwargs...)
    """
    [label]  label[i] == 1 iff the data is classified to class i
    [y]      final input to the loss layer
    """
    # TODO: [label] isn't used
    # local N = size(l.x, 1)
    # if size(l.target,1) != N
    #     l.target    = zeros(size(l.x))
    #     l.dldx = Array{Float64}(size(l.x))
    # end

    # local TAR = zeros(size(l.x))
    # for i = 1:N
    fill!(l.target, 0)
    for i = l.iter
        l.target[i,l.label[i]] = 1
    end

    # local Y = l.x
    # Y = exp(broadcast(+, Y, - maximum(Y,2)))
    l.dldx = exp(l.x)
    broadcast!(/, l.dldx, l.dldx, sum(l.dldx,2))
    broadcast!(-, l.dldx, l.dldx, l.target)
    # return Y .- TAR
    return l.dldx
end

# l = SoftMaxCrossEntropyLoss()
# @time init(l, nothing, Dict{String, Any}("batch_size" => 500, "input_size" => [10]))
# X = rand(500, 10) #input size 784, batch size 500
# L = map(x -> ceil(x), rand(500, 1))
#
# println("First time (compiling...)")
# @time forward(l,X,L)
# @time backward(l,L)
#
# println("Second time ...")
# @time begin
#   for i = 1:10
#     forward(l,X,L)
#   end
# end
# @time begin
#   for i = 1:10
#     backward(l,L)
#   end
# end
#
# println("Third time (profiling...)")
# @time begin
#   for i = 1:1000
#     forward(l,X,L)
#   end
# end
# @time begin
#   for i = 1:1000
#     backward(l,L)
#   end
# end
