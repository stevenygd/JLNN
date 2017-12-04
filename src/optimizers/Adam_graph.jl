type AdamOptimizer
    graph   :: Graph
    m_t     :: Any
    v_t     :: Any
    base_lr :: Float64
    beta_1  :: Float64
    beta_2  :: Float64
    iter    :: Int

    function AdamOptimizer(graph::Graph;  base_lr::Float64=0.001,
                           beta_1::Float64=0.9, beta_2::Float64=0.999)
        m_t, v_t = [], []
        for i = 1:length(graph.forward_order)
           layer = graph.forward_order[i]
           param = getParam(layer)
           if param == nothing
               push!(m_t, nothing)
               push!(v_t, nothing)
           else
               c_1, c_2 = [], []
               for j = 1:length(param)
                   push!(c_1, zeros(size(param[j])))
                   push!(c_2, zeros(size(param[j])))
               end
               push!(m_t, c_1)
               push!(v_t, c_2)
           end;
        end;
        return new(graph, m_t, v_t, base_lr, beta_1, beta_2, 1)
    end
end

function optimize(this::AdamOptimizer, batch_X, batch_Y)

    loss, pred = forward(this.graph, batch_X, batch_Y)
    backward(this.graph, batch_Y)

    for i = 1:length(this.net.layers)
        layer = this.net.layers[i]
        param = getParam(layer)
        if param == nothing
            continue # not a learnable layer
        end

        grad  = getGradient(layer)

        for j = 1:length(param)
            m = this.m_t[i][j]
            v = this.v_t[i][j]
            p = param[j]
            g = grad[j]
            @assert size(m) == size(p) && size(m) == size(g) && size(m) == size(v)

            # Moving average to approximate gradient with velocity
            m = m * this.beta_1 + g    * (1 - this.beta_1)
            v = v * this.beta_2 + g.^2 * (1 - this.beta_2)

            # Compute the counter biased version of [m] and [v]
            m_hat = m / (1. - this.beta_1^this.iter)
            v_hat = v / (1. - this.beta_2^this.iter)

            # Update gradients
            p = p - this.base_lr * m_hat ./ (sqrt.(v_hat) + 1e-4)

            # store the things back
            param[j] = p

            this.m_t[i][j] = m
            this.v_t[i][j] = v
        end
        setParam!(layer, param)
    end

    this.iter += 1;
    return loss, pred
end
