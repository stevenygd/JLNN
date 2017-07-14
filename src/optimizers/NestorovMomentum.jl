type NestorovMomentumOptimizer
    net     :: SequentialNet
    base_lr :: Any
    alpha   :: Float64
    iter    :: Int
    batch_size  ::  Int

    function NestorovMomentumOptimizer(net::SequentialNet, batch_size::Int;
                                       base_lr=(x->0.01), alpha::Float64=0.9)
         return new(net, base_lr, alpha, 1, batch_size)
    end
 end

 function optimize(this::NestorovMomentumOptimizer, batch_X, batch_Y)

     loss, pred = forward(this.net, batch_X, batch_Y)
     backward(this.net, batch_Y)

     for i = 1:length(this.net.layers)
         layer = this.net.layers[i]
         param = getParam(layer)
         if param == nothing
             continue
         end
         gradi = getGradient(layer)
         veloc = getVelocity(layer)

         for j = 1:length(param)
             gradi[j] = this.base_lr(this.iter) * gradi[j] / this.batch_size
             veloc[j] = veloc[j] * this.alpha - gradi[j]
             param[j] = param[j] + this.alpha * veloc[j] - gradi[j]
         end
         setParam!(layer, param)
     end

     this.iter += 1;
     return loss, pred
 end
