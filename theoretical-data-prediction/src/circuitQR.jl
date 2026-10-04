function encode_circuit(nsystem)
    C = QCircuit()
    for i in 1:nsystem
        push!(C,RyGate(i,rand(),isparas=true))
    end
    return C
end

function Quantum_Reservoir_Serial_arrangement(Input_data, memory_qubits, encode_cir, noise_cir, U, B)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys,zs = size(Input_data)
    Results = zeros(length(B),zs)
    for z in 1:zs
        s = StateVector{ComplexF32}(xs+memory_qubits)
        for y in 1:ys
            reset_parameters!(encode_cir,vec(Input_data[:,y:y,z]))
            s = encode_cir * s
            s = U * s
            if noise_cir !== nothing
                s = noise_cir * s
            end
        end
        for (i,b) in enumerate(B)
            Results[i,z] = real(expectation(b,s))[1]
        end
    end
    return Results
end

function Quantum_Reservoir_Serial_arrangement_mul(Input_data, memory_qubits, encode_cir, noise_cir, U, B)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys,zs = size(Input_data)
    Results = zeros(length(B),ys,zs)
    for z in 1:zs
        s = StateVector{ComplexF32}(xs+memory_qubits)
        for y in 1:ys
            reset_parameters!(encode_cir,vec(Input_data[:,y:y,z]))
            s = encode_cir * s
            s = U * s
            if noise_cir !== nothing
                s = noise_cir * s
            end
            for (i,b) in enumerate(B)
                Results[i,y,z] = real(expectation(b,s))[1]
            end
        end
    end
    return Results
end

function train_mul(inputs, y, nmemory, circuit::QCircuit, noise_cir, U, B)
    ndims(inputs) != 3 && error("Input data dimension is not 3!")
    Lb = length(B)
    signal = Quantum_Reservoir_Serial_arrangement_mul(repeat(inputs,1,1,1), nmemory, circuit, noise_cir, U ,B)
    xs,ys,zs = size(inputs)
    xss,yss,zss = size(y)
    Ws=zeros(xss,Lb,ys)
    Es=zeros(ys)
    for j in 1:ys
        Ws[:,:,j] = y[:,j,:]*transpose(signal[:,j,:])*inv(signal[:,j,:]*transpose(signal[:,j,:])+10e-10*Matrix(I,Lb,Lb))
        Es[j]=mean(abs.(Ws[:,:,j]*signal[:,j,:]-y[:,j,:]))
    end
    return Ws, Es
end

function train(inputs, y, nmemory, circuit::QCircuit, noise_cir, U, B)
    ndims(inputs) != 3 && error("Input data dimension is not 3!")
    ndims(y) != 2 && error("Output data y must be 2D, got $(ndims(y))D. Reshape your 3D output before calling train().")
    Lb = length(B)

    signal = Quantum_Reservoir_Serial_arrangement(inputs, nmemory, circuit, noise_cir, U, B)

    W = y * transpose(signal) * inv(signal * transpose(signal) + 10e-7 * Matrix(I, Lb, Lb))

    pred = W * signal
    E = mean(abs.(pred .- y))

    return W, E
end

function Quantum_Reservoir_Serial_arrangement_H(Input_data, memory_qubits, encode_cir, noise_cir, H, tau, B; U=nothing, krylov_threshold=14)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys,zs = size(Input_data)
    nqubit = xs + memory_qubits
    Results = zeros(length(B),zs)
    if U !== nothing
        use_krylov = false
    else
        use_krylov = nqubit > krylov_threshold
        if !use_krylov
            U = convert.(ComplexF32, exp(-im * tau * Matrix(matrix(H))))
        end
    end
    for z in 1:zs
        if use_krylov
            s = StateVector{ComplexF64}(nqubit)
        else
            s = StateVector{ComplexF32}(nqubit)
        end
        for y in 1:ys
            reset_parameters!(encode_cir,vec(Input_data[:,y:y,z]))
            s = encode_cir * s
            if use_krylov
                s = time_evolution(H, -im * tau, s)
            else
                s = U * s
            end
            if noise_cir !== nothing
                s = noise_cir * s
            end
        end
        for (i,b) in enumerate(B)
            Results[i,z] = real(expectation(b,s))[1]
        end
    end
    return Results
end

function train_H(inputs, y, nmemory, circuit::QCircuit, noise_cir, H, tau, B; U=nothing, krylov_threshold=14)
    ndims(inputs) != 3 && error("Input data dimension is not 3!")
    ndims(y) != 2 && error("Output data y must be 2D, got $(ndims(y))D. Reshape your 3D output before calling train_H().")
    Lb = length(B)

    signal = Quantum_Reservoir_Serial_arrangement_H(inputs, nmemory, circuit, noise_cir, H, tau, B; U=U, krylov_threshold=krylov_threshold)

    W = y * transpose(signal) * inv(signal * transpose(signal) + 1e-7 * Matrix(I, Lb, Lb))

    pred = W * signal
    E = mean(abs.(pred .- y))

    return W, E, pred
end
