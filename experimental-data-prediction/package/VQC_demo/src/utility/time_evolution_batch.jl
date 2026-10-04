function time_evolution(h::QubitsOperator, t::Number, v::StateVectorBatch; kwargs...)
	ishermitian(h) || throw(ArgumentError("input operator is not hermitian."))
	(QuantumCircuits.get_largest_pos(h) <= nqubits(v)) || throw(ArgumentError("number of qubits mismatch."))
	n = nqubits(v)
    nt = nitems(v)
	T = promote_type(eltype(h), typeof(t), eltype(v) )
	v = convert(StateVectorBatch{T}, v)
	tmp, info = exponentiate(x -> storage(h(StateVectorBatch(x, n, nt))), t, storage(v); ishermitian=true, kwargs...)
	(info.converged>=1) || error("eigsolve fails to converge.")
	return StateVectorBatch(tmp, n, nt)
end

# function time_evolution(h::QubitsOperator, t::Number, v::StateVectorBatch; kwargs...)
# 	ishermitian(h) || throw(ArgumentError("input operator is not hermitian."))
# 	(QuantumCircuits.get_largest_pos(h) <= nqubits(v)) || throw(ArgumentError("number of qubits mismatch."))
# 	n = nqubits(v)
#     nt = nitems(v)
# 	T = promote_type(eltype(h), typeof(t), eltype(v) )
# 	v = convert(StateVectorBatch{T}, v)
# 	tmp = storage(v)
# 	vout = zeros(T,length(tmp))
# 	for i in 0:100
# 		vout += tmp
# 		tmp = t*storage(h(StateVectorBatch(tmp, n, nt)))./(i+1)
# 	end
# 	return StateVectorBatch(vout, n, nt)
# end