module NN
    export SequentialNet, forward, backward, getGradient,
           getParam, setParam!, getVelocity, getNumParams
    export Layer, DropoutLayer, DenseLayer, ReLu, Sigmoid, SoftMax,
           SoftMaxCrossEntropyLoss, SquareLossLayer, Tanh, InputLayer,
           ConvLayer, MaxPoolingLayer, FlattenLayer,
           CaffeConvLayer

    include("layers/LayerBase.jl")
    include("layers/InputLayer.jl")
    include("layers/DropoutLayer.jl")
    include("layers/DenseLayer.jl")
    include("layers/ReLu.jl")
    include("layers/Sigmoid.jl")
    include("layers/SoftMax.jl")
    include("layers/SoftMaxCrossEntropy.jl")
    include("layers/SquareLossLayer.jl")
    include("layers/Tanh.jl")
    include("layers/SequentialNet.jl")
    include("layers/ConvLayer.jl")
    include("layers/MaxPoolingLayer.jl")
    include("layers/FlattenLayer.jl")
    # include("layers/FlatConvLayer.jl")
    # include("layers/MultiThreadedConvLayer.jl")
    include("layers/CaffeConvLayer.jl")

    # optimizers
    include("optimizers/Adam.jl")
    include("optimizers/RMSprop.jl")
    include("optimizers/Adagrad.jl")
    include("optimizers/NestorovMomentum.jl")
    include("optimizers/SGD.jl")
    include("optimizers/SGDMomentum.jl")
    export AdamOptimizer, RMSPropOptimizer, AdagradOptimizer,
           NestorovMomentumOptimizer, SgdOptimizer, SgdMomentumOptimizer, optimize
end
