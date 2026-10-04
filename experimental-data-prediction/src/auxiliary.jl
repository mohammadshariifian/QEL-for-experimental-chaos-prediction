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
    # 确保输入是非空数组
    if isempty(values)
        throw(ArgumentError("Input array cannot be empty"))
    end

    # 找到最大值
    max_val = maximum(values)

    # 检查最大值是否为零，避免除以零
    if max_val == 0
        throw(ArgumentError("Maximum value of the array is 0, cannot normalize"))
    end

    # 归一化处理
    normalized_values = values ./ max_val

    return normalized_values
end

function relative_entropy(ρ,σ)
    ρ1= storage(ρ)
    σ1 = storage(σ)
    s = tr(ρ1*log(ρ1)) - tr(σ1*log(σ1))
end

function freezen!(link::Vector{Int},H::QubitsOperator)
    for key in keys(H.data)
        for k in key
            if k in link
                delete!(H.data,key)
                break
            end
        end
    end
end

function freezen(link::Vector{Int},H::QubitsOperator)
    H1 = copy(H)
    for key in keys(H1.data)
        for k in key
            if k in link
                delete!(H1.data,key)
                break
            end
        end
    end
    return H1
end


