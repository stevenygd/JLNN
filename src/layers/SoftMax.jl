type SoftMax <: Nonlinearity
    x  :: Array{Float64}
    y :: Array{Float64}
    # last_loss   :: Array{Float64}

    function SoftMax()
        return new(Float64[], Float64[])
    end
end

function forward(l::SoftMax, X::Array{Float64,2}; kwargs...)
	  # X = broadcast(+, X, -maximum(X))
    m,n = size(X)
    Y = zeros(m,n)
    # iterating each row/picture
    for i=1:m
      exp_sum = 0
      # find the exponential sum of output for each class for this row/picture
      for j=1:n
        exp_sum+=exp(X[i,j])
      end
      # softmaxly normalizing the score
      for j=1:n
        Y[i,j]=X[i,j]/exp_sum
      end
    end
    l.y = Y
    return Y
end

function backward(l::SoftMax, DLDY::Array{Float64}; kwargs...)
    @assert size(l.x) == size(DLDY)
    # credits: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network?newreg=d1e89b443dd346ae8bccaf038a944221
    X = l.x
    jacobian = zeros(size(l.x)[1], size(l.x)[1])
    println(typeof(jacobian))
    for i=1:size(l.x)[1]
      for j=1:size(l.x)[1]
        if i==j
          jacobian[i,j] = l.x[j]*(1.0-l.x[j])
        else
          jacobian[i,j] = -l.x[i]*l.x[j]
        end
      end
    end
    return jacobian * DLDY
    # sumX = sum(exp(X))
    # u = zeros(ndims(X), ndims(X))
    # z = zeros(ndims(X))
    # for i = 1: ndims(X)
    #     z[i] = (X[i]/sumX)
    # end
    #
    # for i = 1: ndims(X)
    #     t = z[i]
    #     w = zeros(ndims(X))
    #     w[i] = 1
    #     w = w .- z
    #     u[:,i] = t * w
    # end
    # l.last_loss = DLDY' * u

end
