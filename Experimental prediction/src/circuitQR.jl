
function Circuit_QR_for_Ham_Serial(nmemory, nsystem, p, noisemodel, U, τ)
    nqubit = nsystem + nmemory

    #U = exp(-im*τ*Matrix(matrix(QrH)))
    C = QCircuit()
    # for i in 1:nqubit
    #     push!(C,HGate(i))
    # end
    for i in 1:nsystem
        push!(C,RyGate(i,rand()))
    end

    # G = QuantumGate(Vector(1:nqubit),U)
    # push!(C,G)

    # for i in 1:nqubit
    #     push!(C,noisemodel(i,p=1-exp(-τ*p)))
    # end

    return C
end

function noise_circuit(
    nqubit,
    noisemodel,
    τ,
    γ,
)
    C = QCircuit()

    channel_p = clamp(
        1.0 - exp(
            -4.0 *
            Float64(τ) *
            Float64(γ)
        ),
        0.0,
        1.0,
    )

    for i in 1:nqubit
        push!(
            C,
            noisemodel(
                i;
                p=channel_p,
            ),
        )
    end

    return C
end


function CZ_block!(circuit,ll, p, noisemodel)
    for i in ll
        push!(circuit,RxGate(i,rand()))
        push!(circuit,RzGate(i,rand()))
        push!(circuit,RxGate(i,rand()))
    end
    for i in eachindex(ll)[1:2:end-1]
        push!(circuit,CZGate(ll[i],ll[i+1]))
        push!(circuit,noisemodel(ll[i],p=p))
        push!(circuit,noisemodel(ll[i+1],p=p))
    end
    for i in eachindex(ll)[2:2:end-1]
        push!(circuit,CZGate(ll[i],ll[i+1]))
        push!(circuit,noisemodel(ll[i],p=p))
        push!(circuit,noisemodel(ll[i+1],p=p))
    end
end

function Circuit_QR_XYZ_block!(circuit,ll, p)
    function f!(circuit, k1, k2)
        θx = rand()
        θz = rand()
        θy = rand()
        push!(circuit,RzGate(k2,pi/2,isparas=false))
        push!(circuit,CNOTGate(k2,k1))
        push!(circuit,RzGate(k1,2*θz-pi/2,isparas=false))
        push!(circuit,RyGate(k2,pi/2-2*θy,isparas=false))
        push!(circuit,CNOTGate(k1,k2))
        push!(circuit,RyGate(k2,2*θx-pi/2,isparas=false))
        push!(circuit,CNOTGate(k2,k1))
        push!(circuit,RzGate(k1,-pi/2,isparas=false))
    end
    for i in eachindex(ll)[1:2:end-1]
        f!(circuit,ll[i],ll[i+1])
        push!(circuit,Depolarizing(ll[i],p=p))
        push!(circuit,Depolarizing(ll[i+1],p=p))
    end
    for i in eachindex(ll)[2:2:end-1]
        f!(circuit,ll[i],ll[i+1])
        push!(circuit,Depolarizing(ll[i],p=p))
        push!(circuit,Depolarizing(ll[i+1],p=p))
    end
end



function train(Input_data, circuit::QCircuit, B, memory_qubits, y, U, noise_cir)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    Lb = length(B)
    signal = Quantum_Reservoir_Serial_arrangement(Input_data, circuit, B, memory_qubits, U, noise_cir)
    #signal = Quantum_Reservoir_Parallel_arrangement(Input_data, circuit, B, memory_qubits)
    # if isa(y,Vector)
    #     W = reshape(y,1,:)*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,Lb,Lb))
    # else
    W = y*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,Lb,Lb))
    # end
    return W
end



function apply_local_pauli_decoherence_measurements!(
    Results::AbstractMatrix{<:Real},
    probabilities::AbstractVector{<:Real},
    decoherence_model::Symbol,
)
    nqubits = length(probabilities)

    size(Results, 1) == 3 * nqubits ||
        error(
            "State-vector measurement decoherence requires exactly " *
            "three local observables X,Y,Z per qubit. " *
            "Expected $(3 * nqubits) rows, received $(size(Results, 1))."
        )

    decoherence_model in (:dephasing, :amplitude_damping) ||
        error(
            "decoherence_model must be :dephasing or " *
            ":amplitude_damping. Got $decoherence_model."
        )

    for q in 1:nqubits
        p_q = Float64(probabilities[q])

        isfinite(p_q) && 0.0 <= p_q <= 1.0 ||
            error(
                "Every decoherence probability must be finite " *
                "and lie in [0,1]. Got p[$q]=$p_q."
            )

        x_index = 3 * (q - 1) + 1
        y_index = x_index + 1
        z_index = x_index + 2

        transverse_factor =
            sqrt(max(0.0, 1.0 - p_q))

        @views Results[x_index, :] .*= transverse_factor
        @views Results[y_index, :] .*= transverse_factor

        if decoherence_model === :amplitude_damping
            longitudinal_factor = 1.0 - p_q

            @views Results[z_index, :] .=
                longitudinal_factor .* Results[z_index, :] .+ p_q
        end

        # For dephasing, Z is unchanged.
    end

    return Results
end


function Quantum_Reservoir_Serial_arrangement(
    Input_data,
    circuit,
    B,
    memory_qubits,
    U,
    noise_cir;
    decoherence_backend::Symbol=:full_density_matrix,
    decoherence_probabilities=nothing,
    decoherence_model::Symbol=:dephasing,
)
    ndims(Input_data) == 3 ||
        error(
            "Input data must be three-dimensional. " *
            "Received size $(size(Input_data))."
        )

    nsystem, nsteps, nsamples =
        size(Input_data)

    nqubits =
        nsystem + memory_qubits

    Results =
        zeros(
            Float64,
            length(B),
            nsamples,
        )
    decoherence_backend in (
        :full_density_matrix,
        :statevector_measurement,
    ) ||
        error(
            "decoherence_backend must be :full_density_matrix or " *
            ":statevector_measurement. Got $decoherence_backend."
        )

    use_measurement_decoherence =
        decoherence_backend === :statevector_measurement

    if use_measurement_decoherence
        nsteps == 1 ||
            error(
                "The state-vector measurement method is exact only " *
                "when decoherence is applied after the final evolution. " *
                "This implementation requires size(Input_data,2)==1, " *
                "but received nsteps=$nsteps."
            )

        decoherence_probabilities === nothing &&
            error(
                "decoherence_probabilities must be supplied for " *
                "decoherence_backend=:statevector_measurement."
            )

        length(decoherence_probabilities) == nqubits ||
            error(
                "The probability-vector length must equal nqubits. " *
                "Received $(length(decoherence_probabilities)) values " *
                "for nqubits=$nqubits."
            )

        length(B) == 3 * nqubits ||
            error(
                "The state-vector measurement method currently supports " *
                "only local X,Y,Z measurements. Expected $(3*nqubits) " *
                "observables, received $(length(B))."
            )
    end
    # ========================================================
    # IDEAL RESERVOIR
    #
    # A state vector is sufficient when no quantum channel
    # is applied.
    # ========================================================

    if noise_cir === nothing || use_measurement_decoherence
        for sample_index in 1:nsamples
            ψ =
                StateVector{ComplexF32}(
                    nqubits
                )

            for step_index in 1:nsteps
                input_values =
                    collect(
                        @view Input_data[
                            :,
                            step_index,
                            sample_index,
                        ]
                    )

                all(isfinite, input_values) ||
                    continue

                # Input_data already contains angles in [0, π].
                # Do not multiply by π here.
                reset_parameters!(
                    circuit,
                    input_values,
                )

                ψ =
                    circuit * ψ

                ψ =
                    U * ψ
            end

            for (observable_index, observable) in
                enumerate(B)

                Results[
                    observable_index,
                    sample_index,
                ] =
                    real(
                        expectation(
                            observable,
                            ψ,
                        )
                    )[1]
            end
        end
        if use_measurement_decoherence
            apply_local_pauli_decoherence_measurements!(
                Results,
                decoherence_probabilities,
                decoherence_model,
            )
        end
        return Results
    end

    # ========================================================
    # DECOHERENT RESERVOIR
    #
    # A density matrix is required because phase damping
    # generally produces a mixed state.
    # ========================================================

    for sample_index in 1:nsamples
        ρ =
            DensityMatrix{ComplexF32}(
                nqubits
            )

        for step_index in 1:nsteps
            input_values =
                collect(
                    @view Input_data[
                        :,
                        step_index,
                        sample_index,
                    ]
                )

            all(isfinite, input_values) ||
                continue

            # Input_data is already angle encoded.
            reset_parameters!(
                circuit,
                input_values,
            )

            # Input encoding.
            ρ =
                circuit * ρ



                
            # # Hamiltonian evolution.
            ρ =
                U * ρ * U'

            # # Apply the complete phase-damping circuit.
            ρ =
                noise_cir * ρ
    
        end

        for (observable_index, observable) in
            enumerate(B)

            Results[
                observable_index,
                sample_index,
            ] =
                real(
                    expectation(
                        observable,
                        ρ,
                    )
                )[1]
        end
    end

    return Results
end
