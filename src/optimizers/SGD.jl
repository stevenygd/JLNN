type SgdOptimizer
    net     :: SequentialNet
    base_lr :: Any
    iter    :: Int
    batch_size  ::  Int

    function SgdOptimizer(net::SequentialNet, batch_size;  base_lr=(x->0.01))
         return new(net, base_lr, 1, batch_size)
    end
 end

 function optimize(this::SgdOptimizer, batch_X, batch_Y)

     loss, pred = forward(this.net, batch_X, batch_Y)
     backward(this.net, batch_Y)

     for i = 1:length(this.net.layers)
         layer = this.net.layers[i]

         param = getParam(layer)
         if param == nothing
             continue
         end

         gradi = getGradient(layer)
         for j = 1:length(gradi)
             param[j] -= this.base_lr(this.iter) * gradi[j] / this.batch_size
         end
         setParam!(layer, param)
     end

     this.iter += 1;
     return loss, pred
 end
