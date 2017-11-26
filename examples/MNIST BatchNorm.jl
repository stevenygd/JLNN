include("../src/NN.jl")
using NN
using PyPlot
include("../util/datasets.jl")

X, Y = mnistData(ttl=60000)

function batchnorm_mnist_net(batch_size = 60)
    layers = Layer[
        InputLayer((batch_size, 784))
        DenseLayer(100,init_type="Normal")
        BatchNormLayer()
        ReLu()
        #Sigmoid()
        DenseLayer(100,init_type="Normal")
        BatchNormLayer()
        ReLu()
        #Sigmoid()
        DenseLayer(100,init_type="Normal")
        BatchNormLayer()
        ReLu()
        #Sigmoid()
        DenseLayer(10,init_type="Normal")
    ]
    criteria = SoftMaxCrossEntropyLoss()
    net = SequentialNet(layers, criteria)
    return net
end

function get_corr(pred, answ)
    return length(filter(e -> abs(e) < 1e-5, pred-answ))
end

net = batchnorm_mnist_net()
train_set, test_set, validation_set = datasplit(X, Y)
trX, trY   = train_set
valX, valY = validation_set
teX, teY   = test_set

function train(net::SequentialNet, X, Y; batch_size::Int = 60,
    ttl_epo::Int64 = 10, lrSchedule = (x -> 0.03), verbose=0)
     local N = size(Y)[1]
     local batch = 0
     local epo_losses = []
     local epo_accus = []
     local val_losses = []
     local val_accu   = []
     for epo = 1:ttl_epo
        local num_batch = ceil(N/batch_size)
        if verbose > 5
            println("Epo $(epo) num batches : $(num_batch)")
        end
        all_losses = []
        epo_cor = 0
        for bid = 0:(num_batch-1)
            batch += 1
            local sidx::Int = convert(Int64, bid*batch_size+1)
            local eidx::Int = convert(Int64, min(N, (bid+1)*batch_size))
            local batch_X = X[sidx:eidx,:]
            local batch_Y = Y[sidx:eidx,:]
            loss, _ = forward(net, batch_X, batch_Y)
            backward(net, batch_Y)
            append!(all_losses, mean(loss))
            for i = 1:length(net.layers)
                local layer = net.layers[i]
                param = getParam(layer)
                if param == nothing
                    continue
                end
                local theta
                if i%3!=0 && i<12
                    local gradi = lrSchedule(epo) * getGradient(layer) / batch_size
                    theta = param - gradi
                else # Dealing with BatchNormLayer, have 2 gradients
                    gradi_gamma, gradi_beta = getGradient(layer)
                    gradi_gamma = lrSchedule(epo) * gradi_gamma / batch_size
                    gradi_beta = lrSchedule(epo) * gradi_beta / batch_size
                    theta_gamma = param[1] - gradi_gamma
                    theta_beta = param[2] - gradi_beta
                    theta = Array[theta_gamma,theta_beta]

                end
                setParam!(layer, theta)
            end

            _, pred = forward(net, batch_X, batch_Y; deterministics = true)
            epo_cor  += length(filter(e ->  abs(e) < 1e-5, pred - batch_Y))
            local acc = length(filter(e -> abs(e) < 1e-5, pred - batch_Y)) / batch_size
            if verbose > 1
                println("[$(bid)/$(num_batch)]Loss is: $(loss)\tAccuracy:$(acc)")
            end
        end

        local epo_loss = mean(all_losses)
        local epo_accu = epo_cor / N
        append!(epo_losses, epo_loss)
        append!(epo_accus, epo_accu)
        v_size = size(valX)[1]
        v_loss, v_accu = [],[]
        for i = 1:batch_size:v_size
            curr_v_loss, curr_v_pred = forward(net, valX[i:i+batch_size-1, :], valY[i:i+batch_size-1, :])
            curr_v_accu = get_corr(curr_v_pred, valY[i:i+batch_size-1, :]) / batch_size
            append!(v_loss, curr_v_loss)
            append!(v_accu, curr_v_accu)
        end

        append!(val_losses, mean(v_loss))
        append!(val_accu,   mean(v_accu))
        if verbose > 0
            println("Validation accuracy for epo $(epo): $(v_accu[epo])")
            println("Running validation accuracy: $(mean(v_accu))")
            println("Epo $(epo) has loss :$(mean(epo_loss))\t\taccuracy : $(epo_accu)")
        end
    end
    return epo_losses,epo_accus,val_losses,val_accu
end

losses,accus,val_losses,val_accu = train(net,trX,trY,ttl_epo=20,verbose=1)

figure(figsize=(12,6))
subplot(121)
plot(1:length(losses), losses)
title("Epoch Losses")
show()

# subplot(121)
# plot(1:length(accus), accus)
# title("Epoch Accuracy")


subplot(122)
plot(1: length(val_losses), val_losses)
title("Validation Losses")
show()

# subplot(122)
# plot(1: length(val_accu), val_accu)
# title("Validation Accuracy")
# show()
