include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
import Calculus: check_gradient
using Base.Test

l = SoftMaxCrossEntropyLoss()
function beforeTest(l)
    init(l, nothing, Dict{String, Any}("batch_size" => 1, "input_size" => [3]))
end

function testSoftMaxCrossEntropyOneVector(l, x, y, p, loss, dldx; alpha = 1.)
    beforeTest(l)

    # Testing forwarding
    fl, fp = forward(l,x,y)
    @test_approx_eq fl loss
    @test_approx_eq fp p

    # Testing back propagation
    @test_approx_eq backward(l,y) dldx
end

# First Test
println("Unit test 1...")
x = [1. 2. 3.; -1. -2. -3.]
s = e^(-2) + e^(-1) + 1.
y = [0. 0. 1.; 0. 1. 0.]
p = [2 ; 0]
loss = -log([1./s   ;   e^(-1)/s])

dldx = [ e^(-2)/s e^(-1)/s (1./s - 1); 1./s (e^(-1)/s-1) e^(-2)/s]
testSoftMaxCrossEntropyOneVector(l, x, y, p, loss, dldx)
println("test 1 passed.\n")
