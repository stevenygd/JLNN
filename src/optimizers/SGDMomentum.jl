type SgdMomentumOptimizer
    net     :: SequentialNet
    base_lr :: Any
    iter    :: Int
    batch_size  ::  Int
    mu      :: Float64   # momentum
    function SgdMomentumOptimizer(net::SequentialNet, batch_size;
                                      base_lr=(x->0.01), mu::Float64 = 0.9)
         return new(net, base_lr, 1, batch_size, mu)
    end
 end

 function optimize(this::SgdMomentumOptimizer, batch_X, batch_Y)

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

         for j = 1:length(gradi)
             veloc[j] = this.mu * veloc[j] - this.base_lr(this.iter) * gradi[j] / this.batch_size
             param[j] += veloc[j];  # momentum update
         end
         setParam!(layer, param)
     end

     this.iter += 1;
     return loss, pred
 end
