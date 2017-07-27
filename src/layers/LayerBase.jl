abstract type ANN end
abstract type Layer end
abstract type LearnableLayer <: Layer end
abstract type Nonlinearity   <: Layer end
abstract type LossCriteria   <: Layer end
abstract type DataLayer      <: Layer end
abstract type RegularizationLayer <: Layer end
abstract type UtilityLayer <: Layer end

StaticLayer = Union{Nonlinearity, DataLayer, RegularizationLayer, UtilityLayer}


function getGradient(l::StaticLayer)
    return nothing
end

function getParam(l::StaticLayer)
    return nothing
end

function setParam!(l::StaticLayer, theta)
    nothing
end

function getVelocity(l::StaticLayer)
    return nothing
end

function getInputSize(l::Layer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But input shapes wanted.")
    end
    return size(l.x)
end

function getOutputSize(l::Layer)
    if !l.has_init
        println("Warning: layer $(l) hasn't been initizalized. But output shapes wanted.")
    end
    return size(l.y)
end

function getNumParams(l::Layer)
    return 0
end
