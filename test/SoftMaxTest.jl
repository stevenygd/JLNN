include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMax.jl")
using Base.Test

l = SoftMax();

function testSoftMax(x::Array{Float64}, y::Array{Float64}, dldy::Array{Float64}, dldx::Array{Float64}; err = 0.0001)
  # Testing forwarding
  bools::Array{Bool}
  bools = (forward(l,x)) .- y .<= err
  assertBools(bools)
  bools = l.x .- x .<= err
  assertBools(bools)
  bools = l.y .- y .<= err
  assertBools(bools)

  #Testing back propagation
  bools = backward(l,dldy) .- dldx .<= err
  assertBools(bools)
  # bools = l.dldy .- dldy .<= err
  # assertBools(bools)
  # bools = l.dldx .- dldx .<= err
  # assertBools(bools)
end

function assertBools(bools::Array{Bool})
  for i=1:length(bools)
    @test bools[i]
  end
end

# First Test
println("Unit test 1...")
x = [1.0, 2.0, 3.0]
y = [0.09003, 0.24473, 0.66524]
dldy = [1., 1., 1.]
dldx = [0.00784, 0.00001, 0.00001]
testSoftMax(x, y, dldy, dldx)
println("test 1 passed.\n")
