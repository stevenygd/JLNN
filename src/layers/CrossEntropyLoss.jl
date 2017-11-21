type CrossEntropyLoss <: LossCriteria

  parents  :: Array{Layer}
  children :: Array{Layer}
  has_init :: Bool
  id :: Int64

  x :: Array{Float64}       # input normalized matrix, expected to be free of zeros
  loss :: Array{Float64}    # output of current layer; calculate loss of each instance
  classes :: Int            # number of classes in the model
  dldx :: Array{Float64}    # cache for derivative matrix in backward

  function CrossEntropyLoss()
    return new(
      Layer[], Layer[], false, -1,
      Float64[], Float64[], 0,  Float64[])
  end

  function CrossEntropyLoss(prev::Union{Layer,Void}, config::Union{Dict{String,Any},Void}=nothing)
        layer = new(
          Layer[], Layer[], false, -1,
          Float64[], Float64[], 0,  Float64[])
        init(layer,prev,config)
        layer
    end
end

function init(l::CrossEntropyLoss, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
  if p == nothing
      @assert ndims(config["input_size"]) == 1 # TODO: maybe a error message?
      out_size = (config["batch_size"], config["input_size"][1])
  else
      out_size = getOutputSize(p)
  end
  l.classes = out_size[2]
  l.x      = Array{Float64}(out_size)
  l.loss   = Array{Float64}(out_size[1])
  l.dldx   = Array{Float64}(out_size)
  l.has_init = true
end

function update(l::CrossEntropyLoss, out_size::Tuple)
    l.x = Array{Float64}(out_size)
    l.loss   = Array{Float64}(out_size[1])
    l.dldx = Array{Float64}(out_size)
end

"""
Requried: label needs to be a matrix that assigns a score for each class for each instance
"""
function forward(l::CrossEntropyLoss, Y::Array{Float64,2}, label::Array{Float64, 2}; kwargs...)

  @assert size(Y, 2) == size(l.x, 2)
    m,n = size(Y)
    if m != size(l.x, 1)
      update(l, size(Y))
    end

  l.x = log.(Y)
  l.x = -label.*l.x
  # broadcast!(*, -l.x, label)
  l.loss = sum(l.x,2)
  l.x = Y

  return l.loss, l.x
end

function backward(l::CrossEntropyLoss, label::Array{Float64, 2};kwargs...)
  @assert size(l.x) == size(label)
  l.dldx = -label./l.x
  return l.dldx
end
