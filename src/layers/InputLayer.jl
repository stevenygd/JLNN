# Dummy Layer to create the input shape for initilization
type InputLayer <: DataLayer
    base    :: LayerBase
    dldy    :: Array{Float64}
    shape   :: Tuple

    function InputLayer(shape::Tuple)
        layer = new(LayerBase(), Float64[], shape)
        init(layer, shape)
        layer
    end
end

function init(l::InputLayer, shape::Tuple; kwargs...)
    l.base.y = Array{Float64}(shape)
    l.dldy = Array{Float64}(shape)
end

function update(l::InputLayer, input_size::Tuple)
    # Reinitialize the memory due to the updated of the batch_size
    l.shape = input_size
    l.base.y = Array{Float64}(l.shape)
    l.dldy = Array{Float64}(l.shape)
    # println("Input layer shape update:$(l.shape)")
end

function forward(l::InputLayer, X::Union{SubArray{Float64},Array{Float64}}; kwargs...)
    if size(X) != l.shape
        update(l, size(X))
    end
    l.base.y = X
    return l.base.y
end

function backward(l::InputLayer, DLDY::Array{Float64}; kwargs...)
    l.dldy = DLDY
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
