include("../src/NN.jl")
include("../util/datasets.jl")

using NN
using PyPlot

function build_cnn()
    layers = Layer[
        InputLayer((28,28,1,batch_size)),
        CaffeConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        CaffeConvLayer(32,(5,5)),
        ReLu(),
        MaxPoolingLayer((2,2)),

        FlattenLayer(),

        DenseLayer(256),
        ReLu(),

        DropoutLayer(0.5),
        DenseLayer(10)
    ]

    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end

function OptimizerTest(net::SequentialNet, train_set, validation_set, optimizer;
    batch_size::Int64 = 16, ttl_epo::Int64 = 10)
    X, Y = train_set
    valX, valY = validation_set
    local N = size(Y)[1]
    local batch=0
    local epo_losses = []
    local epo_accus = []

    local val_losses = []
    local val_accu   = []

    iter = 1 # number of iterations
    all_losses = []
    for epo = 1:ttl_epo
        epo_time_used = @elapsed begin
            local num_batch = ceil(N/batch_size)
            epo_cor = 0
            for bid = 0:(num_batch-1)
                time_used = @elapsed begin
                    batch += 1
                    local sidx::Int = convert(Int64, bid*batch_size+1)
                    local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
                    local batch_X = X[:,:,:,sidx:eidx]
                    local batch_Y = Y[sidx:eidx,:]
                    loss, pred = optimize(optimizer, batch_X, batch_Y)
                    append!(all_losses, mean(loss))
                    epo_cor  += get_corr(pred, batch_Y)
                    local acc = get_corr(pred, batch_Y) / batch_size
                end
#                 println("[$(bid)/$(num_batch)]($(time_used)s) Loss is: $(mean(loss))\tAccuracy:$(acc)")
            end

            v_size = size(valX)[1]
            v_loss, v_accu = [],[]
            for i = 1:batch_size:v_size
                batch_X = valX[:,:,:,i:i+batch_size-1]
                batch_Y = valY[i:i+batch_size-1,:]
                curr_v_loss, curr_v_pred = forward(net, batch_X, batch_Y;deterministics=true)
                curr_v_accu = get_corr(curr_v_pred, batch_Y) / batch_size
                append!(v_loss, curr_v_loss)
                append!(v_accu, curr_v_accu)
            end
            append!(val_losses, mean(v_loss))
            append!(val_accu,   mean(v_accu))
        end
        println("Epo $(epo) [$(epo_time_used)s] has loss :$(mean(v_loss))\t\taccuracy : $(mean(v_accu))")
    end
    return epo_losses,epo_accus, val_losses, val_accu,all_losses
end

X,Y = mnistData(ttl=55000) # 0-1
println("X statistics: $(mean(X)) $(minimum(X)) $(maximum(X))")

Y = round(Int, Y)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]
println("TrainSet: $(size(trX)) $(size(trY))")
println("ValSet  : $(size(valX)) $(size(valY))")
println("TestSet : $(size(teX)) $(size(teY))")

trX  = permutedims(reshape(trX,  (size(trX,1),  1, 28, 28)), [3,4,2,1]);
valX = permutedims(reshape(valX, (size(valX,1), 1, 28, 28)), [3,4,2,1]);
teX  = permutedims(reshape(teX,  (size(teX,1),  1, 28, 28)), [3,4,2,1]);

batch_size = 500

net = build_cnn()
# test NestorovMomentum
optimizer = NestorovMomentumOptimizer(net, batch_size,
                                      base_lr=(x->0.01), alpha=0.9)
epo_losses, epo_accu, val_losses, val_accu, Nestorov_losses = OptimizerTest(
    net, (trX, trY), (valX, valY), optimizer;
    ttl_epo = 10, batch_size = batch_size
)
# test SGD
optimizer = SgdOptimizer(net, batch_size, base_lr=(x->0.01))
epo_losses, epo_accu, val_losses, val_accu, Sgd_losses = OptimizerTest(
    net, (trX, trY), (valX, valY), optimizer;
    ttl_epo = 10, batch_size = batch_size
)
# test SGDMomentum
optimizer = SgdMomentumOptimizer(net, batch_size, base_lr=(x->0.01), mu=0.9)
epo_losses, epo_accu, val_losses, val_accu, SgdMomentum_losses = OptimizerTest(
    net, (trX, trY), (valX, valY), optimizer;
    ttl_epo = 10, batch_size = batch_size
  )
# test Adagrad
optimizer = AdagradOptimizer(net, base_lr=(x -> (x<7)?0.01:0.001))
epo_losses, epo_accu, val_losses, val_accu, adagrad_losses = OptimizerTest(
    net, (trX, trY), (valX, valY), optimizer;
    ttl_epo = 10, batch_size = batch_size
)
# test RMSprop
optimizer = RMSPropOptimizer(net, lr_base=0.001, alpha= 0.9)
epo_losses, epo_accu, val_losses, val_accu, RMS_losses = OptimizerTest(
      net, (trX, trY), (valX, valY), optimizer;
      ttl_epo = 10, batch_size = batch_size
  )
# test Adam
optimizer = AdamOptimizer(net, base_lr=0.001,
                       beta_1=0.9, beta_2=0.999)
epo_losses, epo_accu, val_losses, val_accu, adam_losses = OptimizerTest(
    net, (trX, trY), (valX, valY), optimizer;
    ttl_epo = 10, batch_size = batch_size
)

figure(figsize=(12,6))
plot(1:length(adam_losses), adam_losses)
title("Adam : Training losses")
show()
