function (m::QubitsTerm)(vr::StateVectorBatch)
	v = storage(vr)
	vout = similar(v)
	_apply_qterm_util!(m, v, vout,nqubits(vr))
	return StateVectorBatch(vout, nqubits(vr),nitems(vr))
end


function (m::QubitsOperator)(vr::StateVectorBatch) 
	v = storage(vr)
	n = nqubits(vr)
	vout = zeros(eltype(v), length(v))
	if _largest_nterm(m) <= LARGEST_SUPPORTED_NTERMS
		_apply_util!(m, v, vout, n)
	else
		workspace = similar(v)
		for (k, v) in m
			for item in v
			   _apply_qterm_util!(QubitsTerm(k, item), v, workspace, n) 
			   vout .+= workspace
			end
		end
	end
	return StateVectorBatch(vout, nqubits(vr), nitems(vr))
end



Base.:*(m::QubitsOperator,v::StateVectorBatch) = m(v)
Base.:*(m::QubitsTerm,v::StateVectorBatch) =m(v)


Base.:*(m::QubitsTerm,v::DensityMatrixBatch) = m(v)




function (m::QubitsTerm)(vr::DensityMatrixBatch)
	v = vr.data
	vout = similar(v)
	_apply_qterm_util!(m,v,vout,2*nqubits(vr))
	return DensityMatrixBatch(vout, nqubits(vr),nitems(vr))
end


function _apply_qterm_util!(m::QubitsTerm, v::AbstractVector, vout::AbstractVector, n::Int)
	tmp = coeff(m)
	@. vout = tmp * v
	if length(v) >= 32
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_threaded2!(pos, mat, vout, n)
		end	
	else    
		for (pos, mat) in zip(positions(m), oplist(m))
			_apply_gate_2!(pos, mat, vout)
		end			
	end
end


_apply_util!(m::QubitsOperator, v::AbstractVector, vout::AbstractVector, n::Int) = (length(v) >= 32) ? _apply_threaded_util!(
    m, v, vout, n) : _apply_serial_util!(m, v, vout)
