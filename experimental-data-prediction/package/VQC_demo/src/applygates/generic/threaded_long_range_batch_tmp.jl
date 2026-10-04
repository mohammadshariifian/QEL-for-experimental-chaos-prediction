# experimental support for 4-qubit and 5-qubit gate operation

function _apply_fourbody_gate_impl!(key::Tuple{Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4 = key
    pos1, pos2, pos3, pos4 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1)
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(L - 1, 2 * pos4 - 1)

    f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) = begin
        @inbounds for i in ist:ifn
            l0000 = (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-4)<<n
            l1000 = l0000 + posa
            l0100 = l0000 + posb
            l0010 = l0000 + posc
            l0001 = l0000 + posd

            l1100 = l0100 + posa
            l1010 = l0010 + posa
            l1001 = l1000 + posd
            l0110 = l0010 + posb
            l0101 = l0100 + posd
            l0011 = l0010 + posd
            
            l1110 = l1100 + posc
            l1101 = l1100 + posd
            l1011 = l1010 + posd
            l0111 = l0110 + posd

            l1111 = l1110 + posd

            vi = [p[l0000], p[l1000], p[l0100], p[l1100], p[l0010], p[l1010], p[l0110], p[l1110],
                p[l0001], p[l1001], p[l0101], p[l1101], p[l0011], p[l1011], p[l0111], p[l1111]]

            result = mat * vi
            p[l0000], p[l1000], p[l0100], p[l1100], p[l0010], p[l1010], p[l0110], p[l1110],
                p[l0001], p[l1001], p[l0101], p[l1101], p[l0011], p[l1011], p[l0111], p[l1111] = result
        end
    end
    total_itr = div(Ls, 16)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, mask0, mask1, mask2, mask3, mask4, U, v, n)
end

function _apply_sixbody_gate_impl!(key::Tuple{Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4, q5, q6 = key
    pos1, pos2, pos3, pos4, pos5, pos6 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1), 1 << (q6-1)
    
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(pos6 - 1, 2 * pos5 - 1)
    mask6 = xor(L - 1, 2 * pos6 - 1)

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, posf::Int,
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, m7::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) 
        
        # Precompute all possible position combinations
        positions = [0, posa, posb, posc, posd, pose, posf]
        
        @inbounds for i in ist:ifn
            # Calculate base index
            base_idx = (64 * i & m7) | (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-6)<<n
            
            # Generate all 64 indices using bit patterns
            indices = Vector{Int}(undef, 64)
            for j in 0:63
                idx = base_idx
                for k in 1:6
                    if (j & (1 << (k-1))) != 0
                        idx += positions[k+1]
                    end
                end
                indices[j+1] = idx
            end

            # Gather input states
            vi = [p[idx] for idx in indices]

            # Apply matrix multiplication and store results directly
            result = mat * vi
            for (j, idx) in enumerate(indices)
                p[idx] = result[j]
            end
        end
    end

    total_itr = div(Ls, 64)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, pos6, mask0, mask1, mask2, mask3, mask4, mask5, mask6, U, v, n)
end

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_fourbody_gate_impl!(
    key, U, v, n)

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_fivebody_gate_impl!(
    key, U, v, n)

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_sixbody_gate_impl!(
    key, U, v, n)

function _apply_sevenbody_gate_impl!(key::Tuple{Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4, q5, q6, q7 = key
    pos1, pos2, pos3, pos4, pos5, pos6, pos7 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1), 1 << (q6-1), 1 << (q7-1)
    
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(pos6 - 1, 2 * pos5 - 1)
    mask6 = xor(pos7 - 1, 2 * pos6 - 1)
    mask7 = xor(L - 1, 2 * pos7 - 1)

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, posf::Int, posg::Int,
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, m7::Int, m8::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) 
        
        # Precompute all possible position combinations
        positions = [0, posa, posb, posc, posd, pose, posf, posg]
        
        @inbounds for i in ist:ifn
            # Calculate base index
            base_idx = (128 * i & m8) | (64 * i & m7) | (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-7)<<n
            
            # Generate all 128 indices using bit patterns
            indices = Vector{Int}(undef, 128)
            for j in 0:127
                idx = base_idx
                for k in 1:7
                    if (j & (1 << (k-1))) != 0
                        idx += positions[k+1]
                    end
                end
                indices[j+1] = idx
            end

            # Gather input states
            vi = [p[idx] for idx in indices]

            # Apply matrix multiplication and store results directly
            result = mat * vi
            for (j, idx) in enumerate(indices)
                p[idx] = result[j]
            end
        end
    end

    total_itr = div(Ls, 128)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, pos6, pos7, mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, U, v, n)
end

function _apply_eightbody_gate_impl!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4, q5, q6, q7, q8 = key
    pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1), 1 << (q6-1), 1 << (q7-1), 1 << (q8-1)
    
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(pos6 - 1, 2 * pos5 - 1)
    mask6 = xor(pos7 - 1, 2 * pos6 - 1)
    mask7 = xor(pos8 - 1, 2 * pos7 - 1)
    mask8 = xor(L - 1, 2 * pos8 - 1)

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, posf::Int, posg::Int, posh::Int,
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, m7::Int, m8::Int, m9::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) 
        
        # Precompute all possible position combinations
        positions = [0, posa, posb, posc, posd, pose, posf, posg, posh]
        
        @inbounds for i in ist:ifn
            # Calculate base index
            base_idx = (256 * i & m9) | (128 * i & m8) | (64 * i & m7) | (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-8)<<n
            
            # Generate all 256 indices using bit patterns
            indices = Vector{Int}(undef, 256)
            for j in 0:255
                idx = base_idx
                for k in 1:8
                    if (j & (1 << (k-1))) != 0
                        idx += positions[k+1]
                    end
                end
                indices[j+1] = idx
            end

            # Gather input states
            vi = [p[idx] for idx in indices]

            # Apply matrix multiplication and store results directly
            result = mat * vi
            for (j, idx) in enumerate(indices)
                p[idx] = result[j]
            end
        end
    end

    total_itr = div(Ls, 256)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, U, v, n)
end

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_sevenbody_gate_impl!(
    key, U, v, n)

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_eightbody_gate_impl!(
    key, U, v, n)


function _apply_ninebody_gate_impl!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4, q5, q6, q7, q8, q9 = key
    pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1), 1 << (q6-1), 1 << (q7-1), 1 << (q8-1), 1 << (q9-1)
    
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(pos6 - 1, 2 * pos5 - 1)
    mask6 = xor(pos7 - 1, 2 * pos6 - 1)
    mask7 = xor(pos8 - 1, 2 * pos7 - 1)
    mask8 = xor(pos9 - 1, 2 * pos8 - 1)
    mask9 = xor(L - 1, 2 * pos9 - 1)

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, posf::Int, posg::Int, posh::Int, posi::Int,
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, m7::Int, m8::Int, m9::Int, m10::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) 
        
        # Precompute all possible position combinations
        positions = [0, posa, posb, posc, posd, pose, posf, posg, posh, posi]
        
        @inbounds for i in ist:ifn
            # Calculate base index
            base_idx = (512 * i & m10) | (256 * i & m9) | (128 * i & m8) | (64 * i & m7) | (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-9)<<n
            
            # Generate all 512 indices using bit patterns
            indices = Vector{Int}(undef, 512)
            for j in 0:511
                idx = base_idx
                for k in 1:9
                    if (j & (1 << (k-1))) != 0
                        idx += positions[k+1]
                    end
                end
                indices[j+1] = idx
            end

            # Gather input states
            vi = [p[idx] for idx in indices]

            # Apply matrix multiplication and store results directly
            result = mat * vi
            for (j, idx) in enumerate(indices)
                p[idx] = result[j]
            end
        end
    end

    total_itr = div(Ls, 512)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, U, v, n)
end

function _apply_tenbody_gate_impl!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int)
    L = 1<<n
    Ls = length(v)
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = key
    pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10 = 1 << (q1-1), 1 << (q2-1), 1 << (q3-1), 1 << (q4-1), 1 << (q5-1), 1 << (q6-1), 1 << (q7-1), 1 << (q8-1), 1 << (q9-1), 1 << (q10-1)
    
    mask0 = pos1 - 1
    mask1 = xor(pos2 - 1, 2 * pos1 - 1)
    mask2 = xor(pos3 - 1, 2 * pos2 - 1)
    mask3 = xor(pos4 - 1, 2 * pos3 - 1)
    mask4 = xor(pos5 - 1, 2 * pos4 - 1)
    mask5 = xor(pos6 - 1, 2 * pos5 - 1)
    mask6 = xor(pos7 - 1, 2 * pos6 - 1)
    mask7 = xor(pos8 - 1, 2 * pos7 - 1)
    mask8 = xor(pos9 - 1, 2 * pos8 - 1)
    mask9 = xor(pos10 - 1, 2 * pos9 - 1)
    mask10 = xor(L - 1, 2 * pos10 - 1)

    function f(ist::Int, ifn::Int, posa::Int, posb::Int, posc::Int, posd::Int, pose::Int, posf::Int, posg::Int, posh::Int, posi::Int, posj::Int,
        m1::Int, m2::Int, m3::Int, m4::Int, m5::Int, m6::Int, m7::Int, m8::Int, m9::Int, m10::Int, m11::Int, mat::AbstractMatrix, p::AbstractVector, n::Int) 
        
        # Precompute all possible position combinations
        positions = [0, posa, posb, posc, posd, pose, posf, posg, posh, posi, posj]
        
        @inbounds for i in ist:ifn
            # Calculate base index
            base_idx = (1024 * i & m11) | (512 * i & m10) | (256 * i & m9) | (128 * i & m8) | (64 * i & m7) | (32 * i & m6) | (16 * i & m5) | (8 * i & m4) | (4 * i & m3) | (2 * i & m2) | (i & m1) + 1 + i>>(n-10)<<n
            
            # Generate all 1024 indices using bit patterns
            indices = Vector{Int}(undef, 1024)
            for j in 0:1023
                idx = base_idx
                for k in 1:10
                    if (j & (1 << (k-1))) != 0
                        idx += positions[k+1]
                    end
                end
                indices[j+1] = idx
            end

            # Gather input states
            vi = [p[idx] for idx in indices]

            # Apply matrix multiplication and store results directly
            result = mat * vi
            for (j, idx) in enumerate(indices)
                p[idx] = result[j]
            end
        end
    end

    total_itr = div(Ls, 1024)
    parallel_run(total_itr, Threads.nthreads(), f, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, U, v, n)
end

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_ninebody_gate_impl!(
    key, U, v, n)

_apply_gate_threaded2!(key::Tuple{Int, Int, Int, Int, Int, Int, Int, Int, Int, Int}, U::AbstractMatrix, v::AbstractVector, n::Int) = _apply_tenbody_gate_impl!(
    key, U, v, n)
