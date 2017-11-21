# Dummy Layer to create the input shape for initilization
include("LayerBase.jl")
type InputLayer <: DataLayer
    base    :: LayerBase
    shape   :: Tuple
    tag     :: String


    function InputLayer(shape; tag="default")
        # TODO: could allocate less memory by having only two arrays to pass around
        layer = new(Layer(), shape, tag)
        layer.base.has_init = true
        # are below necessary?
        # layer.base.x = Array{Float64}(shape)
        # layer.base.y = Array{Float64}(shape)
        # layer.base.dldy = Array{Float64}(shape)
        layer
    end

    function InputLayer(shape, config::Union{Dict{String,Any},Void}=nothing;tag="default")
        layer = new(LayerBase(), shape, tag)
        layer.base.has_init = true
        # layer.base.x = Array{Float64}(shape)
        # layer.base.y = Array{Float64}(shape)
        # layer.base.dldy = Array{Float64}(shape)
        layer
    end
end

function init(l::InputLayer, p::Union{Layer,Void}, config::Union{Dict{String,Any},Void}; kwargs...)
end

function update(l::InputLayer, input_size::Tuple;)
    # Reinitialize the memory due to the updated of the batch_size
    l.shape = input_size
    # println("Input layer shape update:$(l.shape)")
end

function forward(l::InputLayer, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(X) != l.shape
        update(l, size(X))
    end
    l.base.x = X
    l.base.y = X
    return l.base.y
end

function backward(l::InputLayer; kwargs...)
    l.base.dldy = sum(map(x -> x.base.dldx[l.base.id], l.base.children))
    # l.base.dldy = DLDY
    parent_id = l.base.parents[1].id
    l.base.dldx[parent_id] = l.base.dldy
end

function getInputSize(l::InputLayer)
    return l.shape
end

function getOutputSize(l::InputLayer)
    return l.shape
end

# l = InputLayer((1000,5))
# X = rand(1000, 500)
# Y = rand(1000, 500)
# println("Compile the method for the first time...")
# @time init(l, nothing, Dict{String, Any}("batch_size" => 1000, "input_size" => [500]))
# @time forward(l,X)
# @time backward(l,Y)
#
# println("Start profiling...")
# print("Forward:")
# @time begin
#   for i = 1:10
#     forward(l,X)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end
#
# print("Backward")
# @time begin
#   for i = 1:10
#     backward(l,Y)
#   end
# end
#
# @time begin
#   for i = 1:1000
#     forward(l, X)
#   end
# end
