module NN
    # export SequentialNet, Graph, forward, backward, getGradient,
    #        getParam, setParam!, getVelocity, getNumParams
    # export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
    #        SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
    #        ConvLayer, MaxPoolingLayer, FlattenLayer,
    #        CaffeConvLayer, CrossEntropyLoss
    export Graph, forward, backward, getGradient,
	    getParam, setParam!, getVelocity, getNumParams
    export Layer, DenseLayer, ReLu, DropoutLayer,
	    SoftMaxCrossEntropyLoss, InputLayer
    include("layers/LayerBase.jl")
    include("layers/Graph.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/DenseLayer.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/ReLu.jl")
    # include("layers/Sigmoid.jl")
    # include("layers/SoftMax.jl")
    # include("layers/SquareLossLayer.jl")
    # include("layers/Tanh.jl")
    # include("layers/SequentialNet.jl")
    # include("layers/ConvLayer.jl")
    # include("layers/MaxPoolingLayer.jl")
    # include("layers/FlattenLayer.jl")
    # include("layers/FlatConvLayer.jl")
    # include("layers/MultiThreadedConvLayer.jl")
    # include("layers/CaffeConvLayer.jl")
    # include("layers/CrossEntropyLoss.jl")

    # optimizers
    # include("optimizers/Adam.jl")
    # include("optimizers/AdamPrim.jl")
    # include("optimizers/Bdam.jl")
    # include("optimizers/Cdam.jl")
    # include("optimizers/Ddam.jl")
    # include("optimizers/RMSprop.jl")
    # include("optimizers/SGD.jl")
    include("optimizers/SGD_graph.jl")
    # export AdamOptimizer, AdamPrimOptimizer, BdamOptimizer, CdamOptimizer, DdamOptimizer,
    #        RMSPropOptimizer, SgdOptimizer, optimize
    export SgdOptimizerGraph, optimize
end
