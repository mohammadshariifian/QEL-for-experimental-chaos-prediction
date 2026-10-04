
function set_data!(circuit::QCircuit,data::Matrix{Float64})
    pos=1
    b=size(data,2)
    for gate in circuit
        if typeof(gate)==ERyGate
            gate.paras = data[pos, :]
            gate.batch=(b,)
            pos+=1
        end
    end
end


QuantumReservoirModel(Ham::QubitsOperator, B::Vector{QubitsTerm}, r::Vector{Float64},K::Int,N::Int,M::Int,V::Int,τ::Float64,x::Vector{<:Real}) = QuantumReservoirModel([Ham], B, zeros(length(B),length(r)), r, K, N, M, V, τ, x, sparse(exp(-im*τ*Matrix(matrix(Ham)))), sparse(exp(-im*(τ/V)*Matrix(matrix(Ham)))))

QuantumReservoirModel(circuit::QCircuit, B::Vector{QubitsTerm}, r::Vector{Float64},K::Int,N::Int, M::Int,V::Int,τ::Float64,x::Vector{<:Real}) = QuantumReservoirModel([circuit], B, zeros(length(B),length(r)), r, K, N, M, V, τ, x, zeros(2,2), zeros(2,2))

function partial_trace(state::DensityMatrix, n, N)
	iszero(n) && return state
	return DensityMatrix(partial_trace_optimized(storage(state),n, N), N-n)
end

function partial_trace_optimized(v::AbstractMatrix{<:Complex}, n::Int, N::Int)
    # 检查输入尺寸
    D = size(v, 1)
    @assert size(v, 1) == size(v, 2) "v must be a square matrix."
    @assert 2^N == D "The size of v must be 2^N."
    @assert 0 ≤ n ≤ N "n must satisfy 0 ≤ n ≤ N."

    # 计算维度
    dn = 1 << n          # 2^n
    dr = 1 << (N - n)    # 2^(N-n)

    # 将矩阵 v 视为四维张量： dn × dr × dn × dr
    vt = reshape(v, dn, dr, dn, dr)

    # 分配结果矩阵
    r = zeros(eltype(v), dr, dr)

    # 在前 n 个比特（a=b 的维度）进行部分迹
    @inbounds for a in 1:dn
        r .+= @view vt[a, :, a, :]
    end

    return r
end



function ReservoirOutput_util(U::AbstractMatrix, δU::AbstractMatrix,Input::AbstractArray, B::Vector{QubitsTerm}, V::Int)
    signal = Quantum_Reservoir_util(U, δU, Input, B, V)
    return signal
end




function prediction(Inputs::AbstractArray, circuit::QCircuit, B, W)
    a,b,c = size(Inputs)
    input = reshape(Inputs, a*b, c)
    set_data!(circuit,input)
    #state = StateVectorBatch(nqubit,c)
    state = DensityMatrixBatch(nqubit,c)
    state = circuit * state
    Lb = length(B)
    Output = zeros(Lb, c)
    for i in eachindex(B)
        Output[i,:] = real.(expectation(B[i],state))
    end
    return W*Output
end



# Convenience method: if you ever call Ham_TFIM(nqubit; ...)
Ham_TFIM(nqubit::Int; J::Float64=1.0, h::Float64=2.0, W::Float64=0.05) =
    Ham_TFIM(Random.GLOBAL_RNG, nqubit; J=J, h=h, W=W)
