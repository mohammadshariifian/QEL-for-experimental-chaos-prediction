import LinearAlgebra: log
log(ρ::DensityMatrix) = log(storage(ρ))

import LinearAlgebra: kron
function kron(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

function ⊗(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

import Base: *
function *(A::Matrix{ComplexF64}, B::DensityMatrix{ComplexF64})
    return DensityMatrix(A*storage(B))
end
function *(B::DensityMatrix{ComplexF64},A::Matrix{ComplexF64})
    return DensityMatrix(storage(B)*A)
end
function *(B::DensityMatrix{ComplexF64}, A::Adjoint{ComplexF64, Matrix{ComplexF64}})
    return DensityMatrix(storage(B)*A)
end

function *(A::Matrix, B::DensityMatrix)
    return DensityMatrix(A*storage(B))
end
function *(B::DensityMatrix,A::Adjoint)
    return DensityMatrix(storage(B)*A)
end

function *(A::Matrix,B::StateVector)
    return StateVector(A*storage(B))
end
function *(A::SparseMatrixCSC, B::DensityMatrix)
    return DensityMatrix(A*storage(B))
end

function (c::QCircuit)(p::Vector)
    return reset_parameters!(c,p)
end

function (c::QCircuit)(ρ::DensityMatrix)
    return c*ρ
end

function normalize_to_one(values::Vector{Float64})
    if isempty(values)
        throw(ArgumentError("Input array cannot be empty"))
    end
    max_val = maximum(values)
    if max_val == 0
        throw(ArgumentError("Maximum value of the array is 0, cannot normalize"))
    end
    normalized_values = values ./ max_val
    return normalized_values
end
