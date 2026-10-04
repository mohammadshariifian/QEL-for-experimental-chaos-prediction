function apply!(x::Gate, state::StateVectorBatch) 
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(StateVectorBatch{Complex{T}}, state)
	end
	apply_threaded!(x, storage(state), nqubits(state))
	return state
end

function apply!(x::Gate, state::DensityMatrixBatch)
	@assert _check_pos_range(x, nqubits(state))
	T = eltype(state)
	if (eltype(x) <: Complex) && (T <: Real)
		state = convert(DensityMatrixBatch{Complex{T}}, state)
	end
	_dm_apply_threaded!(x, state.data, nqubits(state))
	return state
end

function apply!(circuit::QCircuit, state::Union{StateVectorBatch, DensityMatrixBatch})
	for gate in circuit
		state = apply!(gate, state)
	end
	return state
end

Base.:*(circuit::QCircuit, state::Union{StateVectorBatch, DensityMatrixBatch}) = apply!(circuit, copy(state))



"""
    currently support mostly 5-qubit gate
"""
apply_threaded!(x::Gate, s::AbstractVector, n::Int) = (length(s) >= 32) ? _apply_gate_threaded2!(ordered_positions(x), ordered_mat(x), s, n) : apply_serial!(x, s)

# unitary gate operation on density matrix, 这里有问题，我改成了适配densitymatrixBatch的类型了，densitymatrix会出现一些问题。
# function _dm_apply_threaded!(x::Gate{N}, s::AbstractVector, n::Int) where N
# 	pos = ordered_positions(x)
# 	m = ordered_mat(x)
# 	if length(s) >= 32
# 		_apply_gate_threaded2!(pos, m, s, 2*n)
# 		_apply_gate_threaded2!(ntuple(i->pos[i]+n, N), conj(m), s, 2*n)
# 	else
# 		_apply_gate_2!(pos, m, s)
# 		_apply_gate_2!(ntuple(i->pos[i]+n, N), conj(m), s)
# 	end
# end

function _dm_apply_threaded!(x::Union{XGate,ZGate,RzGate,CNOTGate,ERyGate}, s::AbstractVector, n::Int)
	pos = ordered_positions(x)
	l=length(pos)
	apply_threaded!(x, s, 2*n)
	pos_tmp=ntuple(i->pos[i]+n, l)
	x_tmp=change_positions(x,Dict(pos[i]=>pos_tmp[i] for i in 1:l))
	apply_threaded!(x_tmp,s,2*n)
end


function _sb_apply_threaded!(x::Gate{N}, s::AbstractVector, n::Int) where N
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 32
		_apply_gate_threaded2!(pos,m,s,n)
	else
		_apply_gate_2!(pos, m, s)
	end
end

#为了更方便地执行密度矩阵的自动微分，效果等价于Uρ

function _dm_apply_threaded_left!(x::Gate, s::AbstractVector, n::Int)
	pos = ordered_positions(x)
	m = ordered_mat(x)
	if length(s) >= 1024
		_apply_gate_threaded2!(pos, m, s, 2*n)
	else
		_apply_gate_2!(pos, m, s)
	end
end

_dm_apply_threaded_left!(x::Union{CNOTGate,XGate,ZGate}, s::AbstractVector, n::Int)=apply_threaded!(x, s, 2*n)


function _dm_apply_threaded!(x::CNOTGate, s::AbstractVector, n::Int)
	pos = ordered_positions(x)
	apply_threaded!(x, s, 2*n)
	pos_tmp=ntuple(i->pos[i]+n, 2)
	x_tmp=change_positions(x,Dict(pos[i]=>pos_tmp[i] for i in 1:2))
	apply_threaded!(x_tmp,s,2*n)
end