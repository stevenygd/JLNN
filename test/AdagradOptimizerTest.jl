
include("../src/NN.jl")
using NN
using PyPlot
using IProfile


function OptimizerTest(net::SequentialNet, train_set, optimizer;
    batch_size::Int64 = 1, ttl_epo::Int64 = 1000)
    batch_X, batch_Y = train_set
    epo_losses = []
    for epo = 1:ttl_epo
        all_losses = []
        for i = 1:4
            X = reshape(batch_X[i, :], 1, 2)
            Y = reshape(batch_Y[i, :], 1, 1)
            loss, pred = optimize(optimizer, X, Y)
            append!(all_losses, loss)
        end
        epo_loss = mean(all_losses)
        append!(epo_losses, epo_loss)
        println("Epo $(epo) has loss:$(epo_loss)")
        println("           layer1: W = $(net.layers[2].W)")
        println("           layer2: W = $(net.layers[4].W)")
    end
    return epo_losses
end

# build the net
layers = Layer[InputLayer((1,2)), DenseLayer(2), Sigmoid(), DenseLayer(1), Sigmoid()]
criteria = SquareLossLayer()
net = SequentialNet(layers, criteria)
# build the layer
w = [0.5 0.5;
     0.5 0.5;]
b = [0. 0;]
W = [w; b;]
setParam!(net.layers[2], Array[W])

w = [ 1.;
      1.;]
b = [0.;]
W = reshape([w; b;], 3, 1)
setParam!(net.layers[4], Array[W])
# training set
trX = [0. 0.;
       0. 1.;
       1. 0.;
       1. 1.;]
trY = [0.; 1.; 0.; 1.]

# test Adagrad
optimizer = AdagradOptimizer(net, base_lr=(x -> (x<7)?0.01:0.001))
epo_losses = OptimizerTest(net, (trX, trY), optimizer, batch_size = 1)

figure(figsize=(12,6))
plot(1:length(epo_losses), epo_losses)
title("Training losses (epoch)")
show()
