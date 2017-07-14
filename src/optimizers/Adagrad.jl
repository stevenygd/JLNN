type AdagradOptimizer
    net     :: SequentialNet
    base_lr :: Any
    cache  :: Any
    iter    :: Int

    function AdagradOptimizer(net::SequentialNet;  base_lr=(x -> (x<7)?0.01:0.001))
        cache = []
        for i = 1:length(net.layers)
            layer = net.layers[i]
            param = getParam(layer)
            if param == nothing
                push!(cache, nothing)
            else
                c = []
                for j = 1:length(param)
                    push!(c, zeros(size(param[j])))
                end
                push!(cache, c)
             end
         end;
         return new(net, base_lr, cache, 1)
     end
 end

 function optimize(this::AdagradOptimizer, batch_X, batch_Y)

     loss, pred = forward(this.net, batch_X, batch_Y)
     backward(this.net, batch_Y)

     for i = 1:length(this.net.layers)
         layer = this.net.layers[i]
         param = getParam(layer)
         if param == nothing
             continue # not a learnable layer
         end

         grad  = getGradient(layer)
         for j = 1:length(param)
             c = this.cache[i][j]
             p = param[j]
             g = grad[j]
             @assert size(c) == size(p) && size(c) == size(g)
             c = c + g.^2
             #                    not sure
             p = p - this.base_lr(this.iter) * g ./ (sqrt(c) + 1e-10)
             this.cache[i][j] = c
             param[j] =    p
         end
         setParam!(layer, param)
     end

     this.iter += 1;
     return loss, pred
 end
