include("../src/NN.jl")
include("../util/datasets.jl")
using NN
using Plots

batch_size = 500

function get_cor(pred, label)
    @assert size(pred) == size(label)
    cor = 0
    for i=1:(size(pred)[1])
        idx = findmax(pred[i, :])[2]
        pred_idx = findmax(label[i, :])[2]
        if idx == pred_idx
            cor += 1
        end
    end
    return cor
end

function multi_sgd(graph::Graph, layerX::Layer, layerY::Layer, optimizer::SgdOptimizerGraph,
    train_set, validation_set;
    batch_size::Int64 = 100, iteration::Int64 = 10, alpha::Float64 = 0.9)

    trX, trY = train_set
    idx = Base.Random.shuffle(1:size(trX)[1])

    for bid = 0:iteration
        local start::Int = bid*batch_size+1
        local last::Int = min(size(trX)[1], (bid+1)*batch_size)
        local batch_X = X[idx[start:last], :]
        local batch_Y = Y[idx[start:last], :]
        optimize(optimizer, Dict(layerX=>batch_X, layerY=>batch_Y))
    end
end

# Data
X,Y = mnistData(ttl=55000)
train_set, test_set, validation_set = datasplit(X,Y;ratio=10./11.)
trX, trY = train_set[1], train_set[2]
valX, valY = validation_set[1], validation_set[2]
teX, teY = test_set[1], test_set[2]

function clone_graph(l1, l3, l5)
    # Clone graph
    thread_layerX = InputLayer((batch_size,784))
    thread_layerY = InputLayer((batch_size, 10))
    thread_l1    = DenseLayer(thread_layerX, 1000; init_type = "Normal")
    thread_l2    = ReLu(thread_l1)
    thread_l3    = DenseLayer(thread_l2, 1000; init_type = "Normal" )
    thread_l4    = ReLu(thread_l3)
    thread_l5    = DenseLayer(thread_l4, 10; init_type = "Normal")
    thread_l6    = SoftMaxCrossEntropyLoss(thread_l5, thread_layerY)
    thread_graph = Graph(thread_l6)
    thread_opt   = SgdOptimizerGraph(thread_graph)
    # Link learnable parameter
    thread_l1.W = l1.W
    thread_l3.W = l3.W
    thread_l5.W = l5.W
    return (thread_layerX, thread_layerY, thread_graph, thread_opt)
end

function benchmark(iter::Int, epo::Int)
    # Creating main graph
    layerX = InputLayer((batch_size,784))
    layerY = InputLayer((batch_size, 10))
    l1    = DenseLayer(layerX, 1000; init_type = "Normal")
    l2    = ReLu(l1)
    l3    = DenseLayer(l2, 1000; init_type = "Normal" )
    l4    = ReLu(l3)
    l5    = DenseLayer(l4, 10; init_type = "Normal")
    l6    = SoftMaxCrossEntropyLoss(l5, layerY)
    graph = Graph(l6)
    # Clone graph for different threads
    clone_array = [clone_graph(l1, l3, l5) for i in 1:iter]
    val_accus = []
    t = @elapsed for e = 1:epo
        Threads.@threads for i = 1:4
            thread_layerX, thread_layerY, thread_graph, thread_opt = clone_array[i]
            # optimize
            multi_sgd(thread_graph, thread_layerX, thread_layerY, thread_opt, (trX, trY), (valX, valY);
            iteration = iter, batch_size = batch_size)
        end

        # println("Running Validation. Validation size: $(size(valY))")
        val_loss, val_pred = forward(graph, Dict(layerX=>valX, layerY=>valY))
        val_loss = mean(val_loss)
        val_size = size(valY)[1]
        val_accu = get_cor(val_pred, valY) / val_size
        push!(val_accus, val_accu)
        println("epo: $(e), iteration: $(iter) validation loss: $(val_loss),
            validation accuracy: $(val_accu)")
    end
    return (t, val_accus[end])
end

times = []
val_accus = []
iters = [i for i = 2:10]
for i in iters
    time_used, val_accu = benchmark(i, 10)
    push!(times, time_used)
    push!(val_accus, val_accu)
end

plotly()
p = plot([times iters iters], [val_accus times val_accus],
        xlabel=["time" "iteration" "iteration"],
        ylabel=["accuracy" "time" "accuracy"],
        label =["accuracy" "time" "accuracy"],
        title = ["accuracy vs. time" "time vs. iteration" "accuracy vs. iteration"],
        window_title = "Mnist on Single Thread",
        marker= (:hexagon, :green),
        size = (1200, 300),
        layout = grid(1, 3))
gui()
