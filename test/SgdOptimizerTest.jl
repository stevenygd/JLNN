include("../src/NN.jl")
include("../src/layers/LayerBase.jl")
include("../src/layers/DenseLayer.jl")
include("../src/layers/Sigmoid.jl")
include("../src/layers/ReLu.jl")
include("../src/layers/SquareLossLayer.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/SequentialNet.jl")
include("../src/layers/InputLayer.jl")

using NN

function OptimizerTest(net::SequentialNet, train_set, optimizer;
    batch_size::Int64 = 4, ttl_epo::Int64 = 1000)

    batch_X, batch_Y = train_set
    batch_X = reshape(batch_X, 4, 2)

    for epo = 1:ttl_epo
          loss, pred = optimize(optimizer, batch_X, batch_Y)
          println("Epo $(epo) has loss :$(loss)\t prediction is $(net.lossfn.x)")
          println("         layer1: W = $(net.layers[2].W)")
          println("         layer2: W = $(net.layers[4].W)")
          #println("         layer2: y = $(net.layers[2].y)")
    end
    return
end
# build the layer
l1 = DenseLayer(2)
init(l1, nothing, Dict{String, Any}("batch_size" => 4, "input_size" => [2]))
w = [0.5 0.5;
     0.5 0.5;]
b = [0. 0;]
W = [w; b;]
setParam!(l1, Array[W])

l2 = DenseLayer(1)
init(l2, l1, Dict{String, Any}("batch_size" => 4, "input_size" => [2]))
w = [ 1.;
      1.;]
b = [0.;]
W = reshape([w; b;], 3, 1)
print(size(l2.W))
setParam!(l2, Array[W])
# training set
trX = [0. 0.;
       0. 1.;
       1. 0.;
       1. 1.;]
trY = [0., 1., 0., 1.]
# build the net
layers = Layer[InputLayer((4,2)), l1, ReLu(), l2, ReLu()]
criteria = SquareLossLayer()
net = SequentialNet(layers, criteria)
# test SGD
optimizer = SgdOptimizer(net, 1 , base_lr=(x->0.01))
OptimizerTest(net, (trX, trY), optimizer, batch_size = 4)
