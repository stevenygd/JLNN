include("../src/layers/LayerBase.jl")
include("../src/layers/SoftMaxCrossEntropy.jl")
include("../src/layers/SoftMax.jl")
include("../src/layers/CrossEntropyLoss.jl")
using Base.Test

smaxcross = SoftMaxCrossEntropyLoss()
smax = SoftMax()
cross = CrossEntropyLoss()


function before(smaxcross, smax, cross, dict)
    init(smaxcross, nothing, dict)
    init(smax     , nothing, dict)
    init(cross    , nothing, dict)
end

function convert_to_one_hot(x::Array{Int64}, classes)
  m = zeros(size(x,1), classes)
  for i=1:size(x,1)
    m[i,x[i]+1]=1
  end
  m
end

function benchmmark(smaxcross, smax, cross, batch_size, input_size)
  # batch_size = 10
  # input_size = 5
  before(smaxcross, smax, cross, Dict{String, Any}("batch_size" => batch_size, "input_size" => [input_size]))
  x = rand(batch_size, input_size)
  label = rand(0:input_size-1, batch_size,1)
  onehot_label = convert_to_one_hot(label,input_size)

  times = zeros(4)
  bytes = zeros(4)

  # benchmark for original SoftMaxCrossEntropy

  times[1] = @elapsed forward(smaxcross,x,onehot_label)
  bytes[1] = @allocated forward(smaxcross,x,onehot_label)

  times[2] = @elapsed backward(smaxcross,onehot_label)
  bytes[2] = @allocated backward(smaxcross,onehot_label)


  # benchmark for newly implemented softmax & CrossEntropyLoss

  times[3] = @elapsed forward(cross, forward(smax,x), onehot_label)
  bytes[3] = @allocated forward(cross, forward(smax,x), onehot_label)

  times[4] = @elapsed forward(cross, forward(smax,x), onehot_label)
  bytes[4] = @allocated forward(cross, forward(smax,x), onehot_label)

  return times, bytes
end

t = zeros(4)
b = zeros(4)

num_iter = 500

for i=1:num_iter
  t_, b_ = benchmmark(smaxcross, smax, cross, 10, 1000)
  t.+= t_
  b.+= b_
end
t/=num_iter
b/=num_iter

println("Old forward uses on average ", t[1], " seconds")
println("New forward uses on average ", t[3], " seconds")
println("Old backward uses on average ", t[2], " seconds")
println("New backward uses on average ", t[4], " seconds")


println("Old forward allocates on average ", b[1], " bytes")
println("New forward allocates on average ", b[3], " bytes")
println("Old backward allocates on average ", b[2], " bytes")
println("New backward allocates on average ", b[4], " bytes")
