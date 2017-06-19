type SoftMax <: Nonlinearity
    x  :: Array{Float64}        # input m by n matrix which represents m instances with scores for n classes
    y :: Array{Float64}         # output m by n matrix which uses softmax to normalize the input matrix
    has_init :: Bool            # true if the layer has been initialized
    jacobian :: Array{Float64}  # cache for the jacobain matrices used in backward
    dldx :: Array{Float64}      # cahce for the backward result

    function SoftMax()
        return new(Float64[], Float64[], false, Float64[], Float64[])
    end
end

function init(l::SoftMax, p::Union{Layer,Void}, config::Dict{String,Any}; kwargs...)
  if p == nothing
      @assert ndims(config["input_size"]) == 1 # TODO: maybe an error message?
      out_size = (config["batch_size"], config["input_size"][1])
  else
      out_size = getOutputSize(p)
  end
  l.x = Array{Float64}(out_size)
  l.y = Array{Float64}(out_size)
  l.has_init = true;
  m,n = out_size
  l.jacobian = Array{Float64}(n,n)
  l.dldx = Array{Float64}(out_size)
end

function forward(l::SoftMax, X::Array{Float64,2}; kwargs...)

    l.x = X
    # iterating each row/picture
    lxexp = exp(l.x)
    lxsum = sum(lxexp, 2)
    l.y = lxexp./lxsum
    return l.y

end

function backward(l::SoftMax, DLDY::Array{Float64}; kwargs...)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    m,n =size(l.x)

    ly = Array{Float64}(n)
    for batch=1:m
      ly = l.y[batch,:]

      l.jacobian .= -ly .* ly'
      l.jacobian[diagind(l.jacobian)] .= ly.*(1.0.-ly)

      # # n x 1 = n x n * n x 1
      l.dldx[batch,:] = l.jacobian * DLDY[batch,:]
    end

    return l.dldx

end
