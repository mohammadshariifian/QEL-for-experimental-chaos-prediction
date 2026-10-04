######################## prediction.jl ########################
# Training:
# - Train the m-horizon-ahead model on CONTEXT only.
# - Use the same input-window construction for open-loop and closed-loop.
#
# Figures:
# - A: RMSE vs FUT time, mean over batches
# - B: Time-series, best batch per feature
# - B2: Time-series with CTX+FUT raw/extrema view
# - C: Time-series, one best batch over all features
#
# Data preparation:
#
# Theory/experiment:
# - Normalize the raw waveform using CTX-only normalization.
# - Remap normalized values to [0, ANGLE_MAX].
##########################################################################################
using Pkg
const PROJECT_ROOT = @__DIR__
Pkg.activate(PROJECT_ROOT)

# Pkg.instantiate()

# Do NOT call Pkg.instantiate() inside the production run script.
# Run it manually only once when setting up the environment:
# julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

using CSV, DataFrames
using Statistics
using Printf
using Random
using LinearAlgebra
using Plots
# gr()
# ENV["GKSwstype"] = "100"

using SpecialFunctions
using JLD2
using Base.Threads
using Plots: mm
using DSP
using DelimitedFiles

const SKIP_IF_RESULT_EXISTS = true
const MAPLIKE = false
const MAPLIKE_TARGET = :value   # :value or :time
const MAPLIKE_WHICH = :min

const USE_MATLAB_PEAKS = MAPLIKE && !Sys.islinux()

if USE_MATLAB_PEAKS
    @eval using MATLAB
    include(joinpath(PROJECT_ROOT, "src", "matlab_peaks.jl"))
else
    function matlab_findpeaks_auto(args...; kwargs...)
        error("MATLAB peak detection is disabled. Set MAPLIKE=false or use Julia peak detection.")
    end

    function extrema_indices_from_normalized(args...; kwargs...)
        error("MATLAB peak detection is disabled. Set MAPLIKE=false or use Julia peak detection.")
    end
end



@info "Threads: $(nthreads())"

include(joinpath(PROJECT_ROOT, "src", "head.jl"))
BLAS.set_num_threads(1)

const DATA_MODE = :theory   # :theory | :experiment


# ============================================================
# TEMPORARY DECOHERENCE DIAGNOSTIC
# ============================================================
const DEBUG_DECOHERENCE_STATE = false
const DEBUG_DECOHERENCE_PRINTED = Ref(false)
const PRINT_ALL_RATE_VECTORS = false
const RATE_PRINT_LOCK = ReentrantLock()
const DEBUG_DECOHERENCE_RATES =
    Ref(Float64[])

const DEBUG_DECOHERENCE_PROBABILITIES =
    Ref(Float64[])

const DEBUG_DECOHERENCE_MODEL =
    Ref(:dephasing)
# -------------------- HAMILTONIAN --------------------
const HAMILTONIAN_KIND = :XYZ_NN
const HAM_NAME = String(HAMILTONIAN_KIND)
const EVOLUTION_TIME_MODE = :tauN   # :tauN or :tau1
# -------------------- CONFIG --------------------
const INPUT_MAPPING_MODE = :qubit   # :qubit or :layer
# CLOSED-loop options
const RUN_CLOSED_LOOP = false
# Only used when RUN_CLOSED_LOOP = true.
# If true, closed-loop prediction resets to true history every `horizon` samples.
# Inside each block, prediction step is always 1.
const RUN_BLOCKWISE_CLOSED_LOOP = false
const TIMESTEPS_PER_QUBIT_LIST = INPUT_MAPPING_MODE == :layer ? (1:1) : (1:11)
const LAYERS_LIST              = INPUT_MAPPING_MODE == :qubit ? (1:1) : (1:5)
const MEMORY_LIST = (0:20)
const MAX_TOTAL_QUBITS = 5
# Input features to include in the model.
const INPUT_FEATURES = ["u1"]

# Available features for theory and experiment:
#   "u1", "u2", "iB", "iC"


const REPEAT_FEATURES = 1
const HORIZON_LIST = MAPLIKE ? [0] : [0]
const DS_LIST = MAPLIKE ? [1] : [25]


const LAG_REP = 1
const feature_size = length(INPUT_FEATURES)
const BASE_NSYSTEM = feature_size * REPEAT_FEATURES

const VS_START = 2.0
const VS_STEP = 0.05
const TARGET_VS = 5.0
const TARGET_RUN_ID = round(Int, (TARGET_VS - VS_START) / VS_STEP) + 1


# ============================================================
# TIME SETTINGS
# ============================================================

# Theory/experiment sampling information
const dt_us_raw = 0.0032
const RAW_SAMPLE_STRIDE = 1

# Theory/experiment intervals, measured in μs
const T_CTX = (5.0, 210.0)
const T_FUT = (210.0, 300.0)

const TIME_UNIT = "μs"

time_axis_label() = "Time ($TIME_UNIT)"

const ACTIVE_T_CTX = T_CTX
const ACTIVE_T_FUT = T_FUT

const PLOT_T_RANGE = (210.0, 230.0)


const BASE_DATA_DIR = normpath(joinpath(PROJECT_ROOT, "data"))

const OUTPUT_DIR = get(
    ENV,
    "QEL_OUTPUT_DIR",
    normpath(joinpath(PROJECT_ROOT, "..", "output")),
)

const FIGURES_DIR   = joinpath(OUTPUT_DIR, "figures")
const RESULTS_DIR   = joinpath(OUTPUT_DIR, "results")
const PRECOMP_U_DIR = joinpath(OUTPUT_DIR, "precomputed_U")


const NORM_MODE = :zscore
const ANGLE_MAX::Float64 = Float64(π)


# reservoir / training hyperparams
const P_noise = 0.0
const λ = 1e-7
lambda_str(λ) = @sprintf("%.0e", λ)
const GAMMA_LIST = [
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,

]

const USE_GAUSSIAN_QUBIT_RATES = true

# Select:
#   :fractional -> σ = GAMMA_STD_FRACTION × mean_rate
#   :fixed      -> σ = GAMMA_FIXED_STD
const GAMMA_STD_MODE = :fractional

# Used only when GAMMA_STD_MODE = :fractional
# 0.01 means 1% of the mean.
# Use 0.10 for 10% of the mean.
const GAMMA_STD_FRACTION = 0.01

# Used only when GAMMA_STD_MODE = :fixed
const GAMMA_FIXED_STD = 0.01

# Fixed seed for reproducible qubit-dependent rates
const GAMMA_PROFILE_SEED = 110

# ============================================================
# DECOHERENCE MODEL SELECTION
#
# :dephasing
#     Lindblad term:
#         γ (ZρZ - ρ)
#     Package probability:
#         p = 1 - exp(-4γτ)
#
# :amplitude_damping
#     Lindblad term:
#         κ (σ⁻ρσ⁺ - 1/2{σ⁺σ⁻,ρ})
#     Package probability:
#         p = 1 - exp(-κτ)
#
# In the sweep below, GAMMA_LIST is used as the rate list.
# For :dephasing, γ means dephasing rate.
# For :amplitude_damping, γ is interpreted as κ.
# ============================================================
# const DECOHERENCE_MODEL = :dephasing
const DECOHERENCE_MODEL = :amplitude_damping

# Select how decoherence is calculated:
#
# :full_density_matrix
#     Complete density-matrix calculation.
#
# :statevector_measurement
#     State-vector evolution followed by analytical modification
#     of the final local X, Y, Z measurements.
const DECOHERENCE_CALCULATION_MODE = :statevector_measurement
# const DECOHERENCE_CALCULATION_MODE = :full_density_matrix


const RESUME_DECOHERENCE = true

# Tolerance used when comparing saved and requested γ values.
const DECOHERENCE_GAMMA_ATOL = 1e-12

# Convert the Lindblad dephasing rate γ to the PhaseDamping
# probability p used by the quantum-circuit package.
#
# We model homogeneous Markovian dephasing as
#
#     dρ/dτ = γ ∑ₖ (Zₖ ρ Zₖ† - ρ)
#
# For one qubit, the off-diagonal coherence obeys
#
#     ρ₀₁(τ) = exp(-2γτ) ρ₀₁(0).
#
# In this package, PhaseDamping(p) attenuates coherences as
#
#     ρ₀₁ -> sqrt(1 - p) ρ₀₁,
#
# which was verified numerically using a |+⟩ state:
#
#     <X> -> sqrt(1 - p) <X>.
#
# Therefore we match both conventions by setting
#
#     sqrt(1 - p) = exp(-2γτ)
#
# so
#
#     p = 1 - exp(-4γτ).
#
# The factor 4 comes only from the package's PhaseDamping(p)
# convention. It is not an extra physical dephasing factor.
# γ = 0 returns `nothing`, meaning no decoherence channel is applied.

function decoherence_channel_type()
    if DECOHERENCE_MODEL === :dephasing
        return PhaseDamping

    elseif DECOHERENCE_MODEL === :amplitude_damping
        return AmplitudeDamping

    else
        error(
            "Invalid DECOHERENCE_MODEL=$DECOHERENCE_MODEL. " *
            "Use :dephasing or :amplitude_damping."
        )
    end
end


function decoherence_model_name()
    if DECOHERENCE_MODEL === :dephasing
        return "PhaseDamping"

    elseif DECOHERENCE_MODEL === :amplitude_damping
        return "AmplitudeDamping"

    else
        error(
            "Invalid DECOHERENCE_MODEL=$DECOHERENCE_MODEL. " *
            "Use :dephasing or :amplitude_damping."
        )
    end
end


function decoherence_probability(
    τ::Real,
    rate::Real,
)
    rate < 0 &&
        error("The decoherence rate must be nonnegative.")

    if DECOHERENCE_MODEL === :dephasing
        # PhaseDamping(p) attenuates coherence as sqrt(1-p).
        # Lindblad dephasing gives ρ01 -> exp(-2γτ)ρ01.
        # Therefore p = 1 - exp(-4γτ).
        return clamp(
            1.0 - exp(-4.0 * Float64(τ) * Float64(rate)),
            0.0,
            1.0,
        )

    elseif DECOHERENCE_MODEL === :amplitude_damping
        # Amplitude damping transfers |1> -> |0>.
        # Excited population decays as exp(-κτ), so
        # p = 1 - exp(-κτ).
        return clamp(
            1.0 - exp(-Float64(τ) * Float64(rate)),
            0.0,
            1.0,
        )

    else
        error(
            "Invalid DECOHERENCE_MODEL=$DECOHERENCE_MODEL. " *
            "Use :dephasing or :amplitude_damping."
        )
    end
end

function gamma_standard_deviation(
    mean_rate::Real,
)
    mean_rate = Float64(mean_rate)

    mean_rate >= 0.0 ||
        error("mean_rate must be nonnegative.")

    # Keep gamma=0 as the exact ideal-reservoir case.
    if mean_rate == 0.0
        return 0.0
    end

    if GAMMA_STD_MODE === :fractional
        return GAMMA_STD_FRACTION * mean_rate

    elseif GAMMA_STD_MODE === :fixed
        return GAMMA_FIXED_STD

    else
        error(
            "Invalid GAMMA_STD_MODE=$GAMMA_STD_MODE. " *
            "Use :fractional or :fixed."
        )
    end
end

function gamma_profile_tag()
    if !USE_GAUSSIAN_QUBIT_RATES
        return "homogeneous"
    end

    if GAMMA_STD_MODE === :fractional
        return @sprintf(
            "gaussianQubit_fractionalStd%.4g_seed%d",
            GAMMA_STD_FRACTION,
            GAMMA_PROFILE_SEED,
        )

    elseif GAMMA_STD_MODE === :fixed
        return @sprintf(
            "gaussianQubit_fixedStd%.4g_seed%d",
            GAMMA_FIXED_STD,
            GAMMA_PROFILE_SEED,
        )

    else
        error(
            "Invalid GAMMA_STD_MODE=$GAMMA_STD_MODE. " *
            "Use :fractional or :fixed."
        )
    end
end

function noise_short_tag()
    DECOHERENCE_MODEL === :dephasing ? "PD" : "AD"
end

function calculation_short_tag()
    DECOHERENCE_CALCULATION_MODE === :statevector_measurement ?
        "SV" : "DM"
end

function profile_short_tag()
    if !USE_GAUSSIAN_QUBIT_RATES
        return "H"
    elseif GAMMA_STD_MODE === :fractional
        return @sprintf(
            "GQf%.4gS%d",
            GAMMA_STD_FRACTION,
            GAMMA_PROFILE_SEED,
        )
    else
        return @sprintf(
            "GQx%.4gS%d",
            GAMMA_FIXED_STD,
            GAMMA_PROFILE_SEED,
        )
    end
end

function sampled_gamma_vector(
    mean_rate::Real,
    nqubit::Int;
    gamma_index::Int,
    batch_index::Int,
    draw_index::Int,
)
    mean_rate = Float64(mean_rate)

    mean_rate < 0 &&
        error("mean_rate must be nonnegative.")

    if !USE_GAUSSIAN_QUBIT_RATES
        return fill(mean_rate, nqubit)
    end

    mean_rate == 0.0 &&
        return zeros(Float64, nqubit)

    # Independent reproducible realization for every:
    # gamma, batch, train/test draw, and qubit.
    rng = MersenneTwister(
        GAMMA_PROFILE_SEED +
        1_000_000 * gamma_index +
        1_000 * batch_index +
        draw_index,
    )

    σ = gamma_standard_deviation(mean_rate)

    rates = Vector{Float64}(undef, nqubit)

    for q in 1:nqubit
        while true
            sampled_rate =
                mean_rate + σ * randn(rng)

            if sampled_rate >= 0.0
                rates[q] = sampled_rate
                break
            end
        end
    end

    return rates
end

function rate_is_nonzero(rate)
    if rate isa AbstractVector
        return any(x -> x > 0, rate)
    else
        return rate > 0
    end
end

function rate_summary(rate)
    if rate isa AbstractVector
        r = Float64.(rate)

        return @sprintf(
            "mean=%.6g, std=%.6g, min=%.6g, max=%.6g",
            mean(r),
            std(r; corrected=false),
            minimum(r),
            maximum(r),
        )
    else
        return @sprintf("%.6g", Float64(rate))
    end
end

function probability_summary(τ::Real, rate)
    if rate isa AbstractVector
        p = [
            decoherence_probability(τ, r)
            for r in Float64.(rate)
        ]

        return @sprintf(
            "mean=%.6g, std=%.6g, min=%.6g, max=%.6g",
            mean(p),
            std(p; corrected=false),
            minimum(p),
            maximum(p),
        )
    else
        return @sprintf("%.6g", decoherence_probability(τ, rate))
    end
end

function make_decoherence_channel(
    nqubit::Int,
    τ::Real,
    rate::Real,
)
    return make_decoherence_channel(
        nqubit,
        τ,
        fill(Float64(rate), nqubit),
    )
end

function make_decoherence_channel(
    nqubit::Int,
    τ::Real,
    rates::AbstractVector{<:Real},
)
    length(rates) == nqubit ||
        error(
            "Length of rates must equal nqubit. " *
            "Got length(rates)=$(length(rates)), nqubit=$nqubit."
        )

    all(r -> isfinite(r) && r >= 0, rates) ||
        error("All decoherence rates must be finite and nonnegative.")

    all(iszero, rates) && return nothing

    channel_type = decoherence_channel_type()

    C = QCircuit()
    has_channel = false

    for q in 1:nqubit
        rate_q = Float64(rates[q])

        rate_q == 0.0 && continue

        channel_p = decoherence_probability(τ, rate_q)

        channel_p == 0.0 && continue

        push!(
            C,
            channel_type(
                q;
                p=channel_p,
            ),
        )

        has_channel = true
    end

    return has_channel ? C : nothing
end




const DECOHERENCE_JOBS = [
    # ========================================================
    # THEORY
    #
    # Theory ds:
    #   u1 = 22
    #   u2 = 8
    #   iB = 35
    #   iC = 69
    # ========================================================

    # -------------------- Theory: 1-to-1 --------------------
    (
        data_mode=:theory,
        config="1to1",
        target="u1",
        features=["u1"],
        ntimesteps=4,
        nmemory=7,
        ds=22,
    ),
    (
        data_mode=:theory,
        config="1to1",
        target="u2",
        features=["u2"],
        ntimesteps=3,
        nmemory=8,
        ds=8,
    ),
    (
        data_mode=:theory,
        config="1to1",
        target="iB",
        features=["iB"],
        ntimesteps=4,
        nmemory=7,
        ds=35,
    ),
    (
        data_mode=:theory,
        config="1to1",
        target="iC",
        features=["iC"],
        ntimesteps=4,
        nmemory=7,
        ds=69,
    ),

    # -------------------- Theory: 3-to-1 --------------------
    (
        data_mode=:theory,
        config="3to1",
        target="u1",
        features=["u1", "iB", "iC"],
        ntimesteps=2,
        nmemory=5,
        ds=22,
    ),

    # No 3-to-1 entry for u2 because u2 is not an input.

    (
        data_mode=:theory,
        config="3to1",
        target="iB",
        features=["u1", "iB", "iC"],
        ntimesteps=2,
        nmemory=5,
        ds=35,
    ),
    (
        data_mode=:theory,
        config="3to1",
        target="iC",
        features=["u1", "iB", "iC"],
        ntimesteps=3,
        nmemory=2,
        ds=69,
    ),

    # -------------------- Theory: 4-to-1 --------------------
    (
        data_mode=:theory,
        config="4to1",
        target="u1",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=1,
        nmemory=7,
        ds=22,
    ),
    (
        data_mode=:theory,
        config="4to1",
        target="u2",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=1,
        nmemory=7,
        ds=8,
    ),
    (
        data_mode=:theory,
        config="4to1",
        target="iB",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=1,
        nmemory=7,
        ds=35,
    ),
    (
        data_mode=:theory,
        config="4to1",
        target="iC",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=2,
        nmemory=3,
        ds=69,
    ),

    # ========================================================
    # EXPERIMENT
    #
    # Experimental ds:
    #   u1 = 25
    #   u2 = 13
    #   iB = 31
    #   iC = 27
    # ========================================================

    # -------------------- Experiment: 1-to-1 --------------------
    (
        data_mode=:experiment,
        config="1to1",
        target="u1",
        features=["u1"],
        ntimesteps=4,
        nmemory=7,
        ds=25,
    ),
    (
        data_mode=:experiment,
        config="1to1",
        target="u2",
        features=["u2"],
        ntimesteps=3,
        nmemory=8,
        ds=13,
    ),
    (
        data_mode=:experiment,
        config="1to1",
        target="iB",
        features=["iB"],
        ntimesteps=4,
        nmemory=7,
        ds=31,
    ),
    (
        data_mode=:experiment,
        config="1to1",
        target="iC",
        features=["iC"],
        ntimesteps=4,
        nmemory=7,
        ds=27,
    ),

    # -------------------- Experiment: 3-to-1 --------------------
    (
        data_mode=:experiment,
        config="3to1",
        target="u1",
        features=["u1", "iB", "iC"],
        ntimesteps=2,
        nmemory=5,
        ds=25,
    ),

    # No 3-to-1 entry for u2 because u2 is not an input.

    (
        data_mode=:experiment,
        config="3to1",
        target="iB",
        features=["u1", "iB", "iC"],
        ntimesteps=2,
        nmemory=5,
        ds=31,
    ),
    (
        data_mode=:experiment,
        config="3to1",
        target="iC",
        features=["u1", "iB", "iC"],
        ntimesteps=2,
        nmemory=5,
        ds=27,
    ),

    # -------------------- Experiment: 4-to-1 --------------------
    (
        data_mode=:experiment,
        config="4to1",
        target="u1",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=2, 
        nmemory=3,
        ds=25,
    ),
    (
        data_mode=:experiment,
        config="4to1",
        target="u2",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=1,
        nmemory=7,
        ds=13,
    ),
    (
        data_mode=:experiment,
        config="4to1",
        target="iB",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=2,
        nmemory=3,
        ds=31,
    ),
    (
        data_mode=:experiment,
        config="4to1",
        target="iC",
        features=["u1", "u2", "iB", "iC"],
        ntimesteps=2,
        nmemory=3,
        ds=27,
    ),
]

const MAX_NEAREST_NEIGHBOR = 1

const APPROX_METHOD = :plain

const BOARD_ID = 1
const N_BATCHES = 100



const MAKE_PREDICTION_FIGURES = true

# XYZ nearest-neighbor coupling sampling
const XYZ_DIST = :uniform
const XYZ_JX_SCALE = 1.0
const XYZ_JY_SCALE = 1.0
const XYZ_JZ_SCALE = 1.0



const OUTPUT320_FILE =
    joinpath(BASE_DATA_DIR, "output_320.csv")




function validate_config!()
    allowed_data_modes = (:theory, :experiment)
    allowed_mapping_modes = (:qubit, :layer)
    allowed_tau_modes = (:tauN, :tau1)
    allowed_norm_modes = (:zscore, :minmax, :global_minmax, :dual_minmax)
    allowed_maplike_targets = (:value, :time)
    allowed_maplike_which = (:max, :min)
    allowed_xyz_dist = (:uniform, :normal)

    DATA_MODE in allowed_data_modes ||
        error(
            "Invalid DATA_MODE=$DATA_MODE. " *
            "Allowed values: $(allowed_data_modes)."
        )

    INPUT_MAPPING_MODE in allowed_mapping_modes ||
        error(
            "Invalid INPUT_MAPPING_MODE=$INPUT_MAPPING_MODE. " *
            "Allowed values: $(allowed_mapping_modes)."
        )

    HAMILTONIAN_KIND === :XYZ_NN ||
        error("Only HAMILTONIAN_KIND=:XYZ_NN is supported.")

    EVOLUTION_TIME_MODE in allowed_tau_modes ||
        error(
            "Invalid EVOLUTION_TIME_MODE=$EVOLUTION_TIME_MODE. " *
            "Allowed values: $(allowed_tau_modes)."
        )

    APPROX_METHOD === :plain ||
        error("Only APPROX_METHOD=:plain is supported.")

    NORM_MODE in allowed_norm_modes ||
        error(
            "Invalid NORM_MODE=$NORM_MODE. " *
            "Allowed values: $(allowed_norm_modes)."
        )

    MAPLIKE_TARGET in allowed_maplike_targets ||
        error(
            "Invalid MAPLIKE_TARGET=$MAPLIKE_TARGET. " *
            "Allowed values: $(allowed_maplike_targets)."
        )

    MAPLIKE_WHICH in allowed_maplike_which ||
        error(
            "Invalid MAPLIKE_WHICH=$MAPLIKE_WHICH. " *
            "Allowed values: $(allowed_maplike_which)."
        )

    XYZ_DIST in allowed_xyz_dist ||
        error(
            "Invalid XYZ_DIST=$XYZ_DIST. " *
            "Allowed values: $(allowed_xyz_dist)."
        )

    GAMMA_STD_MODE in (:fractional, :fixed) ||
        error(
            "GAMMA_STD_MODE must be :fractional or :fixed."
        )

    GAMMA_STD_FRACTION >= 0 ||
        error("GAMMA_STD_FRACTION must be nonnegative.")

    GAMMA_FIXED_STD >= 0 ||
        error("GAMMA_FIXED_STD must be nonnegative.")

    RUN_CLOSED_LOOP isa Bool ||
        error("RUN_CLOSED_LOOP must be true or false.")

    RUN_BLOCKWISE_CLOSED_LOOP isa Bool ||
        error("RUN_BLOCKWISE_CLOSED_LOOP must be true or false.")

    MAPLIKE isa Bool ||
        error("MAPLIKE must be true or false.")

    SKIP_IF_RESULT_EXISTS isa Bool ||
        error("SKIP_IF_RESULT_EXISTS must be true or false.")

    all(x -> x isa Int && x >= 0, MEMORY_LIST) ||
        error("MEMORY_LIST must contain nonnegative integers.")

    all(x -> x isa Int && x > 0, TIMESTEPS_PER_QUBIT_LIST) ||
        error("TIMESTEPS_PER_QUBIT_LIST must contain positive integers.")

    all(x -> x isa Int && x > 0, LAYERS_LIST) ||
        error("LAYERS_LIST must contain positive integers.")

    all(x -> x isa Int && x > 0, DS_LIST) ||
        error("DS_LIST must contain positive integers.")

    all(x -> x isa Int && x >= 0, HORIZON_LIST) ||
        error("HORIZON_LIST must contain nonnegative integers.")

    LAG_REP > 0 ||
        error("LAG_REP must be positive.")

    REPEAT_FEATURES > 0 ||
        error("REPEAT_FEATURES must be positive.")

    MAX_TOTAL_QUBITS >= 2 ||
        error("MAX_TOTAL_QUBITS must be at least 2.")

    N_BATCHES > 0 ||
        error("N_BATCHES must be positive.")



    allowed_features = ("u1", "u2", "iB", "iC")

    all(feature -> feature in allowed_features, INPUT_FEATURES) ||
        error(
            "Invalid INPUT_FEATURES=$INPUT_FEATURES. " *
            "Allowed features are \"u1\", \"u2\", \"iB\", and \"iC\"."
        )

    if DATA_MODE === :theory
        isfile(OUTPUT320_FILE) ||
            error(
                "DATA_MODE=:theory but output_320.csv was not found: " *
                OUTPUT320_FILE
            )

    elseif DATA_MODE === :experiment
        test_file = joinpath(
            BASE_DATA_DIR,
            @sprintf(
                "board_%d_run_%d_new.csv",
                BOARD_ID,
                TARGET_RUN_ID,
            ),
        )

        isfile(test_file) ||
            error(
                "DATA_MODE=:experiment but board CSV was not found: " *
                test_file
            )
    end

    if RUN_CLOSED_LOOP && RUN_BLOCKWISE_CLOSED_LOOP
        all(x -> x > 0, HORIZON_LIST) ||
            error(
                "RUN_BLOCKWISE_CLOSED_LOOP=true requires " *
                "all HORIZON_LIST values to be > 0."
            )
    end

    return nothing
end

validate_config!()


function evolution_time(nqubit::Int)
    EVOLUTION_TIME_MODE in (:tauN, :tau1) ||
        error(
            "EVOLUTION_TIME_MODE must be :tauN or :tau1."
        )

    return EVOLUTION_TIME_MODE === :tauN ?
        Float64(nqubit) :
        1.0
end

    # -------------------- PER-FEATURE FILTER SETTINGS --------------------
const DEFAULT_CUTOFF_HZ   = 5.0e6
const DEFAULT_FILTER_ORDER = 4

const U2_CUTOFF_HZ   = 20.0e6
const U2_FILTER_ORDER = 2

# -------------------- PER-FEATURE PEAK SETTINGS --------------------

const PEAK_SETTINGS_EXPERIMENT = Dict(
    "u1" => (
        max_prominence = 0.0005,
        max_distance   = 30,
        min_prominence = 0.03,
        min_distance   = 30,
    ),
    "u2" => (
        max_prominence = 1.0,
        max_distance   = 100,
        min_prominence = 10.0,
        min_distance   = 100,
    ),
    "iB" => (
        max_prominence = 0.00001,
        max_distance   = 30,
        min_prominence = 0.00001,
        min_distance   = 30,
    ),
    "iC" => (
        max_prominence = 0.0001,
        max_distance   = 30,
        min_prominence = 0.0001,
        min_distance   = 30,
    ),
)

const PEAK_SETTINGS_THEORY = Dict(
    "u1" => (
        max_prominence = 0.0005,
        max_distance   = 30,
        min_prominence = 0.03,
        min_distance   = 30,
    ),
    "u2" => (
        max_prominence = 1.0,
        max_distance   = 100,
        min_prominence = 10.0,
        min_distance   = 100,
    ),
    "iB" => (
        max_prominence = 0.0001,
        max_distance   = 30,
        min_prominence = 0.0001,
        min_distance   = 30,
    ),
    "iC" => (
        max_prominence = 0.0001,
        max_distance   = 30,
        min_prominence = 0.0001,
        min_distance   = 30,
    ),
)

const PEAK_SETTINGS =
    DATA_MODE === :theory ? PEAK_SETTINGS_THEORY : PEAK_SETTINGS_EXPERIMENT
# -------------------- HELPERS --------------------

# -------------------- Normalization helpers --------------------
using Statistics

"""
    normalize_ctx_apply!(ctx, fut; mode=:minmax, input_features=[])

In-place normalization of the context `ctx` and future `fut` matrices (F × T),
returning per-feature offsets/scales `(μ, σ)` so you can later de-normalize.

Supported modes:
- :zscore         -> (x - mean)/std per feature (std floored to 1.0 if 0)
- :minmax         -> (x - min)/(max-min) per feature (range floored to 1.0 if 0)
- :global_minmax  -> min/max computed over all features jointly
- :dual_minmax    -> separate min/max for voltage-like (`"u*"`) vs current-like (`"i*"`) features
"""
function normalize_ctx_apply!(ctx::Matrix{Float64}, fut::Matrix{Float64};
                              mode::Symbol = :minmax,
                              input_features::Vector{String} = String[])
    F = size(ctx, 1)
    μ = zeros(Float64, F)
    σ = ones(Float64,  F)

    if mode === :zscore
        # Use only ctx/training data to compute μ/σ, then apply to both ctx and fut
        @inbounds for f in 1:F
            μ[f] = mean(@view ctx[f, :])
            σ[f] = std(@view ctx[f, :])
            σ[f] = (σ[f] == 0.0) ? 1.0 : σ[f]

            @. ctx[f, :] = (ctx[f, :] - μ[f]) / σ[f]
            if !isempty(fut)
                @. fut[f, :] = (fut[f, :] - μ[f]) / σ[f]
            end
        end


    elseif mode === :minmax
        @inbounds for f in 1:F
            lo = minimum(@view ctx[f, :]); hi = maximum(@view ctx[f, :])
            μ[f] = lo
            σ[f] = (hi - lo == 0.0) ? 1.0 : (hi - lo)
            @. ctx[f, :] = (ctx[f, :] - μ[f]) / σ[f]
            if !isempty(fut)
                @. fut[f, :] = (fut[f, :] - μ[f]) / σ[f]
            end
        end

    elseif mode === :global_minmax
        global_lo = minimum(ctx)
        global_hi = maximum(ctx)
        global_range = (global_hi - global_lo == 0.0) ? 1.0 : (global_hi - global_lo)
        μ .= global_lo
        σ .= global_range
        @. ctx = (ctx - global_lo) / global_range
        if !isempty(fut)
            @. fut = (fut - global_lo) / global_range
        end

    elseif mode === :dual_minmax
        V_IDX = findall(f -> startswith(f, "u"), input_features)
        I_IDX = findall(f -> startswith(f, "i"), input_features)

        if !isempty(V_IDX)
            V_ctx = @view ctx[V_IDX, :]
            V_lo  = minimum(V_ctx); V_hi = maximum(V_ctx)
            V_rg  = (V_hi - V_lo == 0.0) ? 1.0 : (V_hi - V_lo)
            μ[V_IDX] .= V_lo;  σ[V_IDX] .= V_rg
            @. ctx[V_IDX, :] = (ctx[V_IDX, :] - V_lo) / V_rg
            if !isempty(fut)
                @. fut[V_IDX, :] = (fut[V_IDX, :] - V_lo) / V_rg
            end
        end

        if !isempty(I_IDX)
            I_ctx = @view ctx[I_IDX, :]
            I_lo  = minimum(I_ctx); I_hi = maximum(I_ctx)
            I_rg  = (I_hi - I_lo == 0.0) ? 1.0 : (I_hi - I_lo)
            μ[I_IDX] .= I_lo;  σ[I_IDX] .= I_rg
            @. ctx[I_IDX, :] = (ctx[I_IDX, :] - I_lo) / I_rg
            if !isempty(fut)
                @. fut[I_IDX, :] = (fut[I_IDX, :] - I_lo) / I_rg
            end
        end

    else
        error("normalize_ctx_apply!: unknown mode $(mode).")
    end

    return μ, σ
end


# how many distinct lagged times are used in the input window
effective_nlags(mapping_mode::Symbol, ntimesteps::Int, nlayers::Int) =
    mapping_mode === :qubit ? ntimesteps : nlayers

# how many system qubits are needed
# :qubit -> one block per timestep
# :layer -> one block only (same qubits reused across layers)
effective_nsystem(mapping_mode::Symbol, ntimesteps::Int, lag_rep::Int) =
    mapping_mode === :qubit ? (BASE_NSYSTEM * ntimesteps * lag_rep) : BASE_NSYSTEM

# data_file_for(board_id::Integer, run_id::Integer) =
#     joinpath(BASE_DATA_DIR, @sprintf("board_%d_run_%d_new.csv", board_id, run_id))

    
closed_loop_mode_name() =
    RUN_BLOCKWISE_CLOSED_LOOP ? "BLOCKWISE_CLOSED" : "CLOSED"

closed_loop_file_prefix() =
    RUN_BLOCKWISE_CLOSED_LOOP ? "blockwise_closed_loop_only_" : "closed_loop_only_"

closed_loop_label() =
    RUN_BLOCKWISE_CLOSED_LOOP ? "Blockwise closed" : "Closed"



# -------------------- UNIFIED LOAD + NORMALIZE --------------------
function load_and_prepare_data(
    file::String;
    norm=:zscore,
    input_features::Vector{String},
    maplike::Bool=false,
    maplike_which::Symbol=:max,
    maplike_target::Symbol=:value,   # :value or :time
    apply_lowpass::Bool=true,
    default_cutoff_Hz::Real=DEFAULT_CUTOFF_HZ,
    default_filter_order::Int=DEFAULT_FILTER_ORDER,
    u2_cutoff_Hz::Real=U2_CUTOFF_HZ,
    u2_filter_order::Int=U2_FILTER_ORDER,
)
    t_us_all, X_all_raw = read_full_feature_data(file, input_features)

    if apply_lowpass
        X_all_raw = lowpass_filter_matrix(
            X_all_raw, dt_us_raw, input_features;
            default_cutoff_Hz=default_cutoff_Hz,
            default_order=default_filter_order,
            u2_cutoff_Hz=u2_cutoff_Hz,
            u2_order=u2_filter_order,
        )
    end

    ctx_raw, fut_raw, t_ctx_raw, t_fut_raw = split_ctx_fut_from_full(X_all_raw, t_us_all)

    ctx_target_raw = Array{Float64}(ctx_raw)
    fut_target_raw = Array{Float64}(fut_raw)

    μ, σ = normalize_ctx_apply!(ctx_target_raw, fut_target_raw; mode=norm, input_features=input_features)

    ctx_circuit_raw = copy(ctx_target_raw)
    fut_circuit_raw = copy(fut_target_raw)

    μang, σang = remap_ctx_range_to_angle!(ctx_circuit_raw, fut_circuit_raw; angle_max=ANGLE_MAX)

    if !maplike
        return (
            ctx_used = ctx_circuit_raw,
            fut_used = fut_circuit_raw,

            # reservoir target/output scale
            ctx_target = ctx_circuit_raw,
            fut_target = fut_circuit_raw,

            # aligned z-score arrays for RMSE/plots
            ctx_zscore_eval = ctx_target_raw,
            fut_zscore_eval = fut_target_raw,

            # full raw-waveform z-score arrays for diagnostics/CTX+FUT plots
            ctx_zscore_raw = ctx_target_raw,
            fut_zscore_raw = fut_target_raw,

            # diagnostic angle arrays
            ctx_angle_raw = ctx_circuit_raw,
            fut_angle_raw = fut_circuit_raw,

            μ = μ,
            σ = σ,
            μang = μang,
            σang = σang,
            t_axis = t_fut_raw,
            t_ctx_raw = t_ctx_raw,
            t_fut_raw = t_fut_raw,
            last_ctx_extrema_time = fill(NaN, size(ctx_circuit_raw, 1)),
        )
    else
        @assert maplike_which in (:max, :min) "maplike_which must be :max or :min"
        @assert maplike_target in (:value, :time) "maplike_target must be :value or :time"

        if maplike_target === :value
            ctx_map_z, fut_map_z, ctx_map_angle, fut_map_angle, t_axis_map =
                read_segments_maplike_value_aligned(
                    ctx_target_raw,
                    fut_target_raw,
                    ctx_circuit_raw,
                    fut_circuit_raw,
                    t_fut_raw,
                    input_features;
                    which=maplike_which,
                )

            return (
                ctx_used = ctx_map_angle,
                fut_used = fut_map_angle,

                ctx_target = ctx_map_angle,
                fut_target = fut_map_angle,

                ctx_zscore_eval = ctx_map_z,
                fut_zscore_eval = fut_map_z,

                ctx_zscore_raw = ctx_target_raw,
                fut_zscore_raw = fut_target_raw,

                ctx_angle_raw = ctx_circuit_raw,
                fut_angle_raw = fut_circuit_raw,

                μ = μ,
                σ = σ,
                μang = μang,
                σang = σang,
                t_axis = t_axis_map,
                t_ctx_raw = t_ctx_raw,
                t_fut_raw = t_fut_raw,
                last_ctx_extrema_time = fill(NaN, size(ctx_circuit_raw, 1)),
            )
        else
            # MAPLIKE_TARGET === :time
            #
            # Treat the extrema time gaps Δt as the values of a new time series:
            #   Δt_raw -> Δt_zscore -> Δt_angle
            #
            # Reservoir input/target/output: angle-encoded Δt
            # RMSE/plots: z-score Δt

            ctx_gap_raw, fut_gap_raw, t_axis_map, last_ctx_ext_time =
                read_segments_maplike_time_from_normalized(
                    ctx_circuit_raw,
                    fut_circuit_raw,
                    t_ctx_raw,
                    t_fut_raw,
                    input_features;
                    which=maplike_which,
                )

            # z-score-normalize the Δt time series using CTX gaps only
            ctx_gap_zscore = copy(ctx_gap_raw)
            fut_gap_zscore = copy(fut_gap_raw)

            μgap, σgap = normalize_ctx_apply!(
                ctx_gap_zscore,
                fut_gap_zscore;
                mode=:zscore,
                input_features=input_features,
            )

            # map z-score Δt to angle using CTX gap z-score min/max
            ctx_gap_angle = copy(ctx_gap_zscore)
            fut_gap_angle = copy(fut_gap_zscore)

            μgap_ang, σgap_ang = remap_ctx_range_to_angle!(
                ctx_gap_angle,
                fut_gap_angle;
                angle_max=ANGLE_MAX,
            )

            return (
                # reservoir input scale: angle-encoded Δt
                ctx_used = ctx_gap_angle,
                fut_used = fut_gap_angle,

                # reservoir target/output scale: angle-encoded Δt
                ctx_target = ctx_gap_angle,
                fut_target = fut_gap_angle,

                # aligned z-score Δt arrays for RMSE/plots
                ctx_zscore_eval = ctx_gap_zscore,
                fut_zscore_eval = fut_gap_zscore,

                # full waveform z-score arrays for diagnostics/CTX+FUT plots
                ctx_zscore_raw = ctx_target_raw,
                fut_zscore_raw = fut_target_raw,

                # full waveform angle arrays for diagnostics
                ctx_angle_raw = ctx_circuit_raw,
                fut_angle_raw = fut_circuit_raw,

                # original waveform normalization parameters
                μ = μ,
                σ = σ,

                # IMPORTANT:
                # For MAPLIKE_TARGET=:time, these are Δt_zscore -> Δt_angle parameters,
                # not waveform zscore -> waveform angle parameters.
                μang = μgap_ang,
                σang = σgap_ang,

                # optional physical Δt normalization parameters
                μgap = μgap,
                σgap = σgap,

                t_axis = t_axis_map,
                t_ctx_raw = t_ctx_raw,
                t_fut_raw = t_fut_raw,
                last_ctx_extrema_time = last_ctx_ext_time,
            )
        end
    end
end

# -------------------- RAW DATA LOADING / SPLITTING --------------------
function lowpass_filter_matrix(
    X::AbstractMatrix{<:Real},
    dt_us::Real,
    input_features::Vector{String};
    default_cutoff_Hz::Real = DEFAULT_CUTOFF_HZ,
    default_order::Int = DEFAULT_FILTER_ORDER,
    u2_cutoff_Hz::Real = U2_CUTOFF_HZ,
    u2_order::Int = U2_FILTER_ORDER,
)
    fs = 1.0 / (dt_us * 1e-6)   # sampling frequency in Hz

    filt_default = digitalfilter(Lowpass(default_cutoff_Hz), Butterworth(default_order); fs=fs)
    filt_u2      = digitalfilter(Lowpass(u2_cutoff_Hz),      Butterworth(u2_order);      fs=fs)

    Y = Array{Float64}(undef, size(X))

    for f in 1:size(X, 1)
        feat = input_features[f]
        filt = feat == "u2" ? filt_u2 : filt_default
        Y[f, :] = filtfilt(filt, Float64.(X[f, :]))
    end

    return Y
end


function split_ctx_fut_indices(
    t::AbstractVector{<:Real},
)
    ctx_idx = findall(
        x -> T_CTX[1] <= x < T_CTX[2],
        t,
    )

    fut_idx = findall(
        x -> T_FUT[1] <= x < T_FUT[2],
        t,
    )

    if RAW_SAMPLE_STRIDE > 1
        ctx_idx = ctx_idx[1:RAW_SAMPLE_STRIDE:end]
        fut_idx = fut_idx[1:RAW_SAMPLE_STRIDE:end]
    end

    isempty(ctx_idx) &&
        error(
            "No context samples were found in interval $T_CTX."
        )

    isempty(fut_idx) &&
        error(
            "No future samples were found in interval $T_FUT."
        )

    return ctx_idx, fut_idx
end


function split_ctx_fut_from_full(
    X_all::AbstractMatrix,
    t_all::AbstractVector,
)
    ctx_idx, fut_idx = split_ctx_fut_indices(t_all)

    ctx = isempty(ctx_idx) ?
        zeros(Float64, size(X_all, 1), 0) :
        Matrix{Float64}(X_all[:, ctx_idx])

    fut = isempty(fut_idx) ?
        zeros(Float64, size(X_all, 1), 0) :
        Matrix{Float64}(X_all[:, fut_idx])

    t_ctx = isempty(ctx_idx) ?
        Float64[] :
        Float64.(t_all[ctx_idx])

    t_fut = isempty(fut_idx) ?
        Float64[] :
        Float64.(t_all[fut_idx])

    return ctx, fut, t_ctx, t_fut
end

function data_file_for(board_id::Integer, run_id::Integer)
    if DATA_MODE === :theory
        return OUTPUT320_FILE
    elseif DATA_MODE === :experiment
        return joinpath(
            BASE_DATA_DIR,
            @sprintf("board_%d_run_%d_new.csv", board_id, run_id),
        )
    else
        error("DATA_MODE must be :theory or :experiment")
    end
end

function read_output320_feature_data(path::AbstractString, input_features::Vector{String})
    A = readdlm(path, ',', Float64)

    # Remove possible empty rows
    A = A[vec(any(.!isnan.(A), dims=2)), :]

    @assert size(A, 1) >= 4 "output_320.csv must have at least 4 rows: u1, u2, iB, iC"

    rowmap = Dict(
        "u1" => 1,
        "u2" => 2,
        "iB" => 3,
        "iC" => 4,
    )

    for f in input_features
        @assert haskey(rowmap, f) "Feature '$f' is not available in output_320.csv. Use u1, u2, iB, or iC."
    end

    X_all = Matrix{Float64}(A[[rowmap[f] for f in input_features], :])

    # 3.2 ns = 0.0032 μs
    t_us_all = collect(0:size(X_all, 2)-1) .* dt_us_raw

    return t_us_all, X_all
end

function read_board_feature_data(path::AbstractString, input_features::Vector{String})
    df = CSV.read(
        path, DataFrame;
        delim='\t',
        header=false,
        comment="*",
        quotechar='\0',
        ignorerepeated=true,
        types=Float64,
    )
    rename!(df, [:t_s, :u1, :u2, :u3, :iB, :iC])

    t_us_all = Vector(df.t_s .* 1e6)
    X_all = permutedims(hcat([Vector(df[!, Symbol(f)]) for f in input_features]...))
    return t_us_all, Matrix{Float64}(X_all)
end

function read_full_feature_data(
    path::AbstractString,
    input_features::Vector{String},
)
    if DATA_MODE === :theory
        return read_output320_feature_data(
            path,
            input_features,
        )

    elseif DATA_MODE === :experiment
        return read_board_feature_data(
            path,
            input_features,
        )

    else
        error(
            "DATA_MODE must be :theory or :experiment."
        )
    end
end



# -------------------- MAPLIKE EXTRACTION ON NORMALIZED SIGNAL --------------------
function read_segments_maplike_value_aligned(
    ctx_z::AbstractMatrix{<:Real},
    fut_z::AbstractMatrix{<:Real},
    ctx_angle::AbstractMatrix{<:Real},
    fut_angle::AbstractMatrix{<:Real},
    t_fut_raw::AbstractVector{<:Real},
    input_features::Vector{String};
    which::Symbol = :max,
)
    F = size(ctx_z, 1)

    ctx_vals_z      = Vector{Vector{Float64}}(undef, F)
    fut_vals_z      = Vector{Vector{Float64}}(undef, F)
    ctx_vals_angle  = Vector{Vector{Float64}}(undef, F)
    fut_vals_angle  = Vector{Vector{Float64}}(undef, F)
    fut_times       = Vector{Vector{Float64}}(undef, F)

    for f in 1:F
        feat = input_features[f]

        xctx_z = Float64.(vec(ctx_z[f, :]))
        xfut_z = Float64.(vec(fut_z[f, :]))

        xctx_a = Float64.(vec(ctx_angle[f, :]))
        xfut_a = Float64.(vec(fut_angle[f, :]))

        idx_ctx = extrema_indices_from_normalized(xctx_z; which=which, feature_name=feat)
        idx_fut = extrema_indices_from_normalized(xfut_z; which=which, feature_name=feat)

        print_ctx_extrema_variance(feat, xctx_z, idx_ctx, which)

        ctx_vals_z[f]     = isempty(idx_ctx) ? Float64[] : xctx_z[idx_ctx]
        fut_vals_z[f]     = isempty(idx_fut) ? Float64[] : xfut_z[idx_fut]
        ctx_vals_angle[f] = isempty(idx_ctx) ? Float64[] : xctx_a[idx_ctx]
        fut_vals_angle[f] = isempty(idx_fut) ? Float64[] : xfut_a[idx_fut]
        fut_times[f]      = isempty(idx_fut) ? Float64[] : Float64.(t_fut_raw[idx_fut])
    end

    NctxE = minimum(length.(ctx_vals_z))
    NfutE = minimum(length.(fut_vals_z))

    ctx_z_out = NctxE == 0 ? zeros(Float64, F, 0) :
        permutedims(hcat([v[1:NctxE] for v in ctx_vals_z]...))

    fut_z_out = NfutE == 0 ? zeros(Float64, F, 0) :
        permutedims(hcat([v[1:NfutE] for v in fut_vals_z]...))

    ctx_angle_out = NctxE == 0 ? zeros(Float64, F, 0) :
        permutedims(hcat([v[1:NctxE] for v in ctx_vals_angle]...))

    fut_angle_out = NfutE == 0 ? zeros(Float64, F, 0) :
        permutedims(hcat([v[1:NfutE] for v in fut_vals_angle]...))

    t_axis_fut = NfutE == 0 ? Float64[] : fut_times[1][1:NfutE]

    return ctx_z_out, fut_z_out, ctx_angle_out, fut_angle_out, t_axis_fut
end

function print_ctx_extrema_variance(feat::String, x_ctx::Vector{Float64}, idx_ctx::Vector{Int}, which::Symbol)
    vals = isempty(idx_ctx) ? Float64[] : Float64.(x_ctx[idx_ctx])

    if isempty(vals)
        println(@sprintf(
            "CTX extrema variance | feature=%s | which=%s | count=0 | variance=NaN",
            feat, String(which)
        ))
    elseif length(vals) == 1
        println(@sprintf(
            "CTX extrema variance | feature=%s | which=%s | count=1 | variance=0.000000e+00",
            feat, String(which)
        ))
    else
        println(@sprintf(
            "CTX extrema variance | feature=%s | which=%s | count=%d | variance=%.6e",
            feat, String(which), length(vals), var(vals)
        ))
    end
end

function auto_prominence_from_signal(x::Vector{Float64})
    if length(x) < 5
        return 0.0
    end
    dx = diff(x)
    med_dx = median(dx)
    mad_dx = median(abs.(dx .- med_dx))
    sigma = 1.4826 * mad_dx
    rng = maximum(x) - minimum(x) + eps()
    return max(0.02 * rng, 3.0 * sigma)
end




function read_segments_maplike_time_from_normalized(
    ctx_norm_raw::AbstractMatrix{<:Real},
    fut_norm_raw::AbstractMatrix{<:Real},
    t_ctx_raw::AbstractVector{<:Real},
    t_fut_raw::AbstractVector{<:Real},
    input_features::Vector{String};
    which::Symbol = :max,
)
    F = size(ctx_norm_raw, 1)

    ctx_gaps  = Vector{Vector{Float64}}(undef, F)
    fut_gaps  = Vector{Vector{Float64}}(undef, F)
    fut_tpos  = Vector{Vector{Float64}}(undef, F)
    last_ctx_extrema_time = Vector{Float64}(undef, F)

    for f in 1:F
        feat = input_features[f]

        x_ctx = Float64.(vec(ctx_norm_raw[f, :]))
        x_fut = Float64.(vec(fut_norm_raw[f, :]))
        tctx  = Float64.(t_ctx_raw)
        tfut  = Float64.(t_fut_raw)

        idx_ctx = extrema_indices_from_normalized(x_ctx; which=which, feature_name=feat)
        idx_fut = extrema_indices_from_normalized(x_fut; which=which, feature_name=feat)

        print_ctx_extrema_variance(feat, x_ctx, idx_ctx, which)

        t_ctx_ext = isempty(idx_ctx) ? Float64[] : tctx[idx_ctx]
        t_fut_ext = isempty(idx_fut) ? Float64[] : tfut[idx_fut]

        last_ctx_time = isempty(t_ctx_ext) ? NaN : t_ctx_ext[end]
        last_ctx_extrema_time[f] = last_ctx_time

        ctx_gaps[f] = length(t_ctx_ext) >= 2 ? diff(t_ctx_ext) : Float64[]

        if isfinite(last_ctx_time) && !isempty(t_fut_ext)
            # include bridge gap: last CTX extremum -> first FUT extremum
            fut_gaps[f] = diff(vcat(last_ctx_time, t_fut_ext))
            fut_tpos[f] = copy(t_fut_ext)
        else
            fut_gaps[f] = Float64[]
            fut_tpos[f] = Float64[]
        end
    end

    Nctx = minimum(length.(ctx_gaps))
    Nfut = minimum(length.(fut_gaps))

    ctx = Nctx == 0 ? zeros(Float64, F, 0) :
          permutedims(hcat([v[1:Nctx] for v in ctx_gaps]...))

    fut = Nfut == 0 ? zeros(Float64, F, 0) :
          permutedims(hcat([v[1:Nfut] for v in fut_gaps]...))

    t_axis_fut = Nfut == 0 ? Float64[] : fut_tpos[1][1:Nfut]

    return ctx, fut, t_axis_fut, last_ctx_extrema_time
end

# -------------------- UNIFIED LOAD + NORMALIZE --------------------

function remap_ctx_range_to_angle!(ctx::Matrix{Float64}, fut::Matrix{Float64};
                                   angle_max::Float64 = ANGLE_MAX)
    F = size(ctx, 1)

    # Compute global min and max across all features
    global_lo = minimum(ctx)
    global_hi = maximum(ctx)
    global_range = (global_hi - global_lo == 0.0) ? 1.0 : (global_hi - global_lo)

    @inbounds for f in 1:F
        @. ctx[f, :] = angle_max * (ctx[f, :] - global_lo) / global_range
        if !isempty(fut)
            @. fut[f, :] = angle_max * (fut[f, :] - global_lo) / global_range
        end
    end

    # return same offset/scale per feature for compatibility
    offset = fill(global_lo, F)
    scale = fill(global_range, F)
    return offset, scale
end

function circuit_angle_to_zscore(
    θ::AbstractVector{<:Real},
    μang::AbstractVector{<:Real},
    σang::AbstractVector{<:Real};
    angle_max::Float64 = ANGLE_MAX,
)
    return (Float64.(θ) ./ angle_max) .* σang .+ μang
end

function circuit_angle_to_zscore_matrix(
    Θ::AbstractMatrix{<:Real},
    μang::AbstractVector{<:Real},
    σang::AbstractVector{<:Real};
    angle_max::Float64 = ANGLE_MAX,
)
    Z = Array{Float64}(undef, size(Θ))
    @inbounds for s in axes(Θ, 2)
        Z[:, s] .= circuit_angle_to_zscore(@view(Θ[:, s]), μang, σang; angle_max=angle_max)
    end
    return Z
end

function circuit_angle_to_zscore_tensor(
    Θ::AbstractArray{<:Real,3},
    μang::AbstractVector{<:Real},
    σang::AbstractVector{<:Real};
    angle_max::Float64=ANGLE_MAX,
)
    F, T, Q = size(Θ)

    X = Array{Float64}(undef, F, T, Q)

    @inbounds for q in 1:Q
        @views X[:, :, q] .=
            circuit_angle_to_zscore_matrix(
                Θ[:, :, q],
                μang,
                σang;
                angle_max=angle_max,
            )
    end

    return X
end

function repeat_features_matrix(X::AbstractMatrix{<:Real}, repeats::Int)
    if repeats == 1
        return Array{Float64}(X)
    end
    if size(X, 2) == 0
        return zeros(Float64, size(X, 1) * repeats, 0)
    end
    return Array{Float64}(repeat(X, repeats, 1))
end


function gaps_to_times(gaps::AbstractVector{<:Real}, t0::Real)
    out = Array{Float64}(undef, length(gaps))
    acc = Float64(t0)
    @inbounds for i in eachindex(gaps)
        acc += Float64(gaps[i])
        out[i] = acc
    end
    return out
end



function build_windows_ctx_offset_dual(
    X_input::AbstractMatrix{<:Real},
    X_target::AbstractMatrix{<:Real};
    ntimesteps::Int,
    nlayers::Int,
    ds::Int,
    lag_rep::Int,
    ahead_target::Int,
    mapping_mode::Symbol,
)
    inp, _ = build_windows_ctx_offset(
        X_input;
        ntimesteps=ntimesteps,
        nlayers=nlayers,
        ds=ds,
        lag_rep=lag_rep,
        ahead_target=ahead_target,
        mapping_mode=mapping_mode,
    )

    _, y = build_windows_ctx_offset(
        X_target;
        ntimesteps=ntimesteps,
        nlayers=nlayers,
        ds=ds,
        lag_rep=lag_rep,
        ahead_target=ahead_target,
        mapping_mode=mapping_mode,
    )

    return inp, y
end
# ---------- m-horizon training windows on CONTEXT (for OPEN m-horizon) ----------
# inputs: X[:, k0 + (0:(nlayers-1))*ds]
# target: X[:, k0 + nlayers*ds + ahead_target]
function build_windows_ctx_offset(
    X::AbstractMatrix{<:Real};
    ntimesteps::Int,
    nlayers::Int,
    ds::Int,
    lag_rep::Int,
    ahead_target::Int,
    mapping_mode::Symbol,
)
    F, N = size(X)

    nlags = effective_nlags(mapping_mode, ntimesteps, nlayers)

    max_k0 = N - nlags * ds - ahead_target
    if max_k0 < 1
        if mapping_mode === :qubit
            return zeros(Float64, F * ntimesteps * lag_rep, 1, 0), zeros(Float64, F, 0)
        else
            return zeros(Float64, F, nlayers * lag_rep, 0), zeros(Float64, F, 0)
        end
    end

    T = max_k0
    y = Array{Float64}(undef, F, T)

    if mapping_mode === :qubit
        inp = Array{Float64}(undef, F * ntimesteps * lag_rep, 1, T)

        @inbounds for (k, k0) in enumerate(1:T)
            target_idx = k0 + nlags * ds + ahead_target
            @views y[:, k] .= X[:, target_idx]

            for step_base in 1:ntimesteps
                lag_idx = k0 + (step_base - 1) * ds
                for r in 1:lag_rep
                    s_eff = (step_base - 1) * lag_rep + r
                    row_lo = (s_eff - 1) * F + 1
                    row_hi = s_eff * F
                    @views inp[row_lo:row_hi, 1, k] .= X[:, lag_idx]
                end
            end
        end

    elseif mapping_mode === :layer
        inp = Array{Float64}(undef, F, nlayers * lag_rep, T)

        @inbounds for (k, k0) in enumerate(1:T)
            target_idx = k0 + nlags * ds + ahead_target
            @views y[:, k] .= X[:, target_idx]

            for layer_base in 1:nlayers
                lag_idx = k0 + (layer_base - 1) * ds
                for r in 1:lag_rep
                    s_eff = (layer_base - 1) * lag_rep + r
                    @views inp[:, s_eff, k] .= X[:, lag_idx]
                end
            end
        end

    else
        error("INPUT_MAPPING_MODE must be :qubit or :layer")
    end

    return inp, y
end

# --------- build B ---------
function build_B_ops(input_features::Vector{String}, nsystem::Int, nmemory::Int)
    base_feats = input_features
    F0 = length(base_feats)
    @assert nsystem % F0 == 0 "nsystem must be a multiple of length(INPUT_FEATURES)"
    repeats = div(nsystem, F0)
    nqubit = nsystem + nmemory

    B = QubitsTerm[]

    # system qubits
    for r in 0:(repeats-1)
        for (j, feat) in enumerate(base_feats)
            q = r * F0 + j
            primary = "X"
            push!(B, QubitsTerm(q => primary))
            for op in ("X", "Y", "Z")
                if op != primary
                    push!(B, QubitsTerm(q => op))
                end
            end
        end
    end

    # memory qubits
    for q in (nsystem+1):nqubit
        for op in ("X", "Y", "Z")
            push!(B, QubitsTerm(q => op))
        end
    end

    # k-body
    paulis = ("X", "Y", "Z")
    for k in 2:MAX_NEAREST_NEIGHBOR
        for q in 1:(nqubit-(k-1))
            for combo in Base.Iterators.product(ntuple(_ -> paulis, k)...)
                ops = ((q + i) => combo[i+1] for i in 0:(k-1))
                push!(B, QubitsTerm(ops...))
            end
        end
    end

    return B
end

# ----------------- OPEN-LOOP (teacher-forced over FUT) -----------------

function compute_open_teacher_forced_for_batch(
    fp::Int,
    ahead::Int,
    ctx_circuit::AbstractMatrix{<:Real},
    fut_circuit::AbstractMatrix{<:Real},
    ctx_target::AbstractMatrix{<:Real},
    fut_target::AbstractMatrix{<:Real},
    nsystem::Int,
    nmemory::Int,
    ntimesteps::Int,
    nlayers::Int,
    Us::Vector{<:AbstractMatrix},
    τ::Float64,
    ds::Int,
    B1::AbstractVector;
    γ_train=0.0,
    γ_test=0.0,
    lag_rep::Int=LAG_REP,
    mapping_mode::Symbol=INPUT_MAPPING_MODE,
    train_inputs_override=nothing,
    train_targets_override=nothing,
    decoherence_backend::Symbol=:full_density_matrix,
)
    F, N_ctx = size(ctx_circuit)
    _, N_fut = size(fut_circuit)

    nlags = effective_nlags(mapping_mode, ntimesteps, nlayers)

    if N_fut <= 0
        return zeros(Float64, F, 0)
    end

    U = Us[fp]
    circuit = Circuit_QR_for_Ham_Serial(
        nmemory,
        nsystem,
        P_noise,
        decoherence_channel_type(),
        U,
        τ,
    )

    nqubit = nsystem + nmemory

    decoherence_backend in (
        :full_density_matrix,
        :statevector_measurement,
    ) ||
        error(
            "decoherence_backend must be :full_density_matrix or " *
            ":statevector_measurement. Got $decoherence_backend."
        )

    train_rates =
        γ_train isa AbstractVector ?
        Float64.(γ_train) :
        fill(Float64(γ_train), nqubit)

    test_rates =
        γ_test isa AbstractVector ?
        Float64.(γ_test) :
        fill(Float64(γ_test), nqubit)

    length(train_rates) == nqubit ||
        error(
            "Length of γ_train must equal nqubit=$nqubit. " *
            "Received $(length(train_rates))."
        )

    length(test_rates) == nqubit ||
        error(
            "Length of γ_test must equal nqubit=$nqubit. " *
            "Received $(length(test_rates))."
        )

    train_probabilities = [
        decoherence_probability(τ, rate_q)
        for rate_q in train_rates
    ]

    test_probabilities = [
        decoherence_probability(τ, rate_q)
        for rate_q in test_rates
    ]

    train_noise_cir =
        decoherence_backend === :full_density_matrix ?
        make_decoherence_channel(
            nqubit,
            τ,
            train_rates,
        ) :
        nothing

    test_noise_cir =
        decoherence_backend === :full_density_matrix ?
        make_decoherence_channel(
            nqubit,
            τ,
            test_rates,
        ) :
        nothing

    # ---- Train OPEN m-horizon model on CONTEXT ----
    if (train_inputs_override === nothing) !=
    (train_targets_override === nothing)

        error(
            "train_inputs_override and train_targets_override " *
            "must either both be provided or both be nothing."
        )
    end

    train_inputs_open, train_Y_open =
        if train_inputs_override === nothing
            build_windows_ctx_offset_dual(
                ctx_circuit,
                ctx_target;
                ntimesteps=ntimesteps,
                nlayers=nlayers,
                ds=ds,
                lag_rep=lag_rep,
                ahead_target=ahead,
                mapping_mode=mapping_mode,
            )
        else
            (
                train_inputs_override,
                train_targets_override,
            )
        end

    if size(train_Y_open, 2) == 0
        return fill(NaN, F, N_fut)
    end

    R_train_open = Quantum_Reservoir_Serial_arrangement(
        train_inputs_open,
        circuit,
        B1,
        nmemory,
        U,
        train_noise_cir;
        decoherence_backend=decoherence_backend,
        decoherence_probabilities=train_probabilities,
        decoherence_model=DECOHERENCE_MODEL,
    )

    G_open = Hermitian(
        R_train_open * R_train_open' + λ * I
    )

    W_open = (
        cholesky(G_open) \
        (R_train_open * train_Y_open')
    )'

    # ---- Teacher-forced prediction over FUT ----
    X_all = hcat(ctx_circuit, fut_circuit)
    N_all = size(X_all, 2)

    if mapping_mode === :qubit
        inp = Array{Float64}(undef, F * ntimesteps * lag_rep, 1, N_fut)
    else
        inp = Array{Float64}(undef, F, nlayers * lag_rep, N_fut)
    end

    @inbounds for s in 1:N_fut
        target_idx = N_ctx + s
        k0 = target_idx - nlags * ds - ahead

        if k0 < 1
            @views inp[:, :, s] .= NaN
            continue
        end

        bad_window = false

        if mapping_mode === :qubit
            for step_base in 1:ntimesteps
                lag_idx = k0 + (step_base - 1) * ds
                if lag_idx < 1 || lag_idx > N_all
                    bad_window = true
                    break
                end

                for r in 1:lag_rep
                    s_eff = (step_base - 1) * lag_rep + r
                    row_lo = (s_eff - 1) * F + 1
                    row_hi = s_eff * F
                    @views inp[row_lo:row_hi, 1, s] .= X_all[:, lag_idx]
                end
            end

        elseif mapping_mode === :layer
            for layer_base in 1:nlayers
                lag_idx = k0 + (layer_base - 1) * ds
                if lag_idx < 1 || lag_idx > N_all
                    bad_window = true
                    break
                end

                for r in 1:lag_rep
                    s_eff = (layer_base - 1) * lag_rep + r
                    @views inp[:, s_eff, s] .= X_all[:, lag_idx]
                end
            end
        else
            error("INPUT_MAPPING_MODE must be :qubit or :layer")
        end

        if bad_window
            @views inp[:, :, s] .= NaN
        end
    end

    R_open = Quantum_Reservoir_Serial_arrangement(
        inp,
        circuit,
        B1,
        nmemory,
        U,
        test_noise_cir;
        decoherence_backend=decoherence_backend,
        decoherence_probabilities=test_probabilities,
        decoherence_model=DECOHERENCE_MODEL,
    )
    if DEBUG_DECOHERENCE_STATE &&
        fp == 1 &&
        rate_is_nonzero(γ_test)
        R_open_without_noise =
            Quantum_Reservoir_Serial_arrangement(
                inp,
                circuit,
                B1,
                nmemory,
                U,
                nothing,
            )

        println("γ_test = ", rate_summary(γ_test))
        println("τ = ", τ)
        println("decoherence model = ", decoherence_model_name())
        println("channel p = ", probability_summary(τ, γ_test))
        println("max |R_noisy - R_ideal| = ",
            maximum(abs.(R_open .- R_open_without_noise))
        )
        println("relative change = ",
            norm(R_open .- R_open_without_noise) / norm(R_open_without_noise)
        )
    end
    Y_open = W_open * R_open

    if DEBUG_DECOHERENCE_STATE &&
        fp == 1 &&
        rate_is_nonzero(γ_test)
        R_open_without_noise =
            Quantum_Reservoir_Serial_arrangement(
                inp,
                circuit,
                B1,
                nmemory,
                U,
                nothing,
            )

        Y_open_without_noise = W_open * R_open_without_noise

        println("γ_train = ", rate_summary(γ_train))
        println("γ_test  = ", rate_summary(γ_test))
        println("τ = ", τ)
        println("decoherence model = ", decoherence_model_name())
        println("channel p = ", probability_summary(τ, γ_test))
        
        println("relative reservoir change = ",
            norm(R_open .- R_open_without_noise) / norm(R_open_without_noise)
        )

        println("relative prediction change = ",
            norm(Y_open .- Y_open_without_noise) / norm(Y_open_without_noise)
        )

        println("max |Y_noisy - Y_ideal| = ",
            maximum(abs.(Y_open .- Y_open_without_noise))
        )
    end

    yhat_open = Array{Float64}(undef, F, N_fut)
    @views yhat_open[:, 1:N_fut] .= Y_open[:, 1:N_fut]
    return yhat_open
end

# ----------------- CLOSED-LOOP over FUT -----------------
function compute_closed_loop_for_batch(
    fp::Int,
    ahead::Int,
    ctx_circuit::AbstractMatrix{<:Real},
    fut_circuit::AbstractMatrix{<:Real},
    ctx_target::AbstractMatrix{<:Real},
    fut_target::AbstractMatrix{<:Real},
    μang_circuit::AbstractVector{<:Real},
    σang_circuit::AbstractVector{<:Real},
    nsystem::Int,
    nmemory::Int,
    ntimesteps::Int,
    nlayers::Int,
    Us::Vector{<:AbstractMatrix},
    τ::Float64,
    ds::Int,
    B1::AbstractVector;
    lag_rep::Int=LAG_REP,
    mapping_mode::Symbol=INPUT_MAPPING_MODE,
)
    F, N_ctx = size(ctx_circuit)
    _, N_fut = size(fut_circuit)

    nlags = effective_nlags(mapping_mode, ntimesteps, nlayers)

    if N_fut <= 0
        return zeros(Float64, F, 0)
    end

    U = Us[fp]
    circuit = Circuit_QR_for_Ham_Serial(
        nmemory,
        nsystem,
        P_noise,
        decoherence_channel_type(),
        U,
        τ,
    )
    noise_cir = nothing

    # Train same m-horizon model on CONTEXT only
    train_inputs, train_Y = build_windows_ctx_offset_dual(
        ctx_circuit,
        ctx_target;
        ntimesteps=ntimesteps,
        nlayers=nlayers,
        ds=ds,
        lag_rep=lag_rep,
        ahead_target=ahead,
        mapping_mode=mapping_mode,
    )

    if size(train_Y, 2) == 0
        return fill(NaN, F, N_fut)
    end

    R_train = Quantum_Reservoir_Serial_arrangement(
        train_inputs, circuit, B1, nmemory, U, noise_cir
    )

    G = R_train * R_train' + λ * I
    W = train_Y * R_train' * inv(G)

    # Storage for recursive FUT predictions
    yhat_closed = fill(NaN, F, N_fut)

    for s in 1:N_fut
        target_idx = N_ctx + s
        k0 = target_idx - nlags * ds - ahead

        if k0 < 1
            continue
        end

        if mapping_mode === :qubit
            inp_s = Array{Float64}(undef, F * ntimesteps * lag_rep, 1, 1)
        else
            inp_s = Array{Float64}(undef, F, nlayers * lag_rep, 1)
        end

        bad_window = false

        if mapping_mode === :qubit
            for step_base in 1:ntimesteps
                lag_idx = k0 + (step_base - 1) * ds

                if lag_idx < 1 || lag_idx > N_ctx + N_fut
                    bad_window = true
                    break
                end

                xlag = if lag_idx <= N_ctx
                    ctx_circuit[:, lag_idx]
                else
                    fut_s = lag_idx - N_ctx

                    # CLOSED LOOP:
                    # If this lag lies in FUT, use the already predicted value.
                    # If it has not been predicted yet, this target cannot be computed.
                    if fut_s < 1 || fut_s >= s || any(.!isfinite.(yhat_closed[:, fut_s]))
                        bad_window = true
                        break
                    end

                    @view(yhat_closed[:, fut_s])
                end

                for r in 1:lag_rep
                    s_eff = (step_base - 1) * lag_rep + r
                    row_lo = (s_eff - 1) * F + 1
                    row_hi = s_eff * F
                    @views inp_s[row_lo:row_hi, 1, 1] .= xlag
                end
            end

        elseif mapping_mode === :layer
            for layer_base in 1:nlayers
                lag_idx = k0 + (layer_base - 1) * ds

                if lag_idx < 1 || lag_idx > N_ctx + N_fut
                    bad_window = true
                    break
                end

                xlag = if lag_idx <= N_ctx
                    ctx_circuit[:, lag_idx]
                else
                    fut_s = lag_idx - N_ctx

                    if fut_s < 1 || fut_s >= s || any(.!isfinite.(yhat_closed[:, fut_s]))
                        bad_window = true
                        break
                    end

                    @view(yhat_closed[:, fut_s])
                end

                for r in 1:lag_rep
                    s_eff = (layer_base - 1) * lag_rep + r
                    @views inp_s[:, s_eff, 1] .= xlag
                end
            end

        else
            error("INPUT_MAPPING_MODE must be :qubit or :layer")
        end

        if bad_window
            continue
        end

        R_s = Quantum_Reservoir_Serial_arrangement(inp_s, circuit, B1, nmemory, U, noise_cir)
        Y_s = W * R_s
        @views yhat_closed[:, s] .= Y_s[:, 1]
    end

    return yhat_closed
end


# ----------------- BLOCKWISE CLOSED-LOOP over FUT -----------------
function compute_blockwise_closed_loop_for_batch(
    fp::Int,
    ahead::Int,
    ctx_circuit::AbstractMatrix{<:Real},
    fut_circuit::AbstractMatrix{<:Real},
    ctx_target::AbstractMatrix{<:Real},
    fut_target::AbstractMatrix{<:Real},
    μang_circuit::AbstractVector{<:Real},
    σang_circuit::AbstractVector{<:Real},
    nsystem::Int,
    nmemory::Int,
    ntimesteps::Int,
    nlayers::Int,
    Us::Vector{<:AbstractMatrix},
    τ::Float64,
    ds::Int,
    B1::AbstractVector;
    lag_rep::Int=LAG_REP,
    mapping_mode::Symbol=INPUT_MAPPING_MODE,
)
    if ahead <= 0
        error("RUN_BLOCKWISE_CLOSED_LOOP=true requires horizon > 0 because horizon is used as the block size.")
    end

    F, N_ctx = size(ctx_circuit)
    _, N_fut = size(fut_circuit)

    nlags = effective_nlags(mapping_mode, ntimesteps, nlayers)

    if N_fut <= 0
        return zeros(Float64, F, 0)
    end

    U = Us[fp]
    circuit = Circuit_QR_for_Ham_Serial(
        nmemory,
        nsystem,
        P_noise,
        decoherence_channel_type(),
        U,
        τ,
    )
    noise_cir = nothing

    # In blockwise mode, train using consecutive 1-step lags.
    # The horizon `ahead` is still used as the target offset / block size.
    train_inputs, train_Y = build_windows_ctx_offset_dual(
        ctx_circuit,
        ctx_target;
        ntimesteps=ntimesteps,
        nlayers=nlayers,
        ds=1,
        lag_rep=lag_rep,
        ahead_target=ahead,
        mapping_mode=mapping_mode,
    )

    if size(train_Y, 2) == 0
        return fill(NaN, F, N_fut)
    end

    R_train = Quantum_Reservoir_Serial_arrangement(
        train_inputs, circuit, B1, nmemory, U, noise_cir
    )

    G = R_train * R_train' + λ * I
    W = train_Y * R_train' * inv(G)

    yhat_closed = fill(NaN, F, N_fut)

    # True full timeline is used only to seed each block.
    X_true = hcat(ctx_circuit, fut_circuit)
    N_all = N_ctx + N_fut

    block_size = ahead
    block_start_s = 1

    while block_start_s <= N_fut
        block_end_s = min(block_start_s + block_size - 1, N_fut)

        for s in block_start_s:block_end_s
            target_idx = N_ctx + s

            # IMPORTANT:
            # In blockwise mode, the rollout step is always 1.
            # Therefore inside a block we predict consecutive FUT points:
            # s, s+1, s+2, ...
            #
            # The model itself is still trained using `ahead`.
            k0 = target_idx - nlags

            if k0 < 1
                continue
            end

            if mapping_mode === :qubit
                inp_s = Array{Float64}(undef, F * ntimesteps * lag_rep, 1, 1)
            else
                inp_s = Array{Float64}(undef, F, nlayers * lag_rep, 1)
            end

            bad_window = false

            if mapping_mode === :qubit
                for step_base in 1:ntimesteps
                    lag_idx = k0 + (step_base - 1)

                    if lag_idx < 1 || lag_idx > N_all
                        bad_window = true
                        break
                    end

                    xlag = if lag_idx <= N_ctx
                        ctx_circuit[:, lag_idx]
                    else
                        fut_s = lag_idx - N_ctx

                        if fut_s < block_start_s
                            # Before this block: reset to truth.
                            X_true[:, lag_idx]
                        elseif fut_s < s && isfinite(fut_s) && all(isfinite.(yhat_closed[:, fut_s]))
                            # Inside this block: use previous prediction.
                            @view(yhat_closed[:, fut_s])
                        else
                            bad_window = true
                            break
                        end
                    end

                    for r in 1:lag_rep
                        s_eff = (step_base - 1) * lag_rep + r
                        row_lo = (s_eff - 1) * F + 1
                        row_hi = s_eff * F
                        @views inp_s[row_lo:row_hi, 1, 1] .= xlag
                    end
                end

            elseif mapping_mode === :layer
                for layer_base in 1:nlayers
                    lag_idx = k0 + (layer_base - 1)

                    if lag_idx < 1 || lag_idx > N_all
                        bad_window = true
                        break
                    end

                    xlag = if lag_idx <= N_ctx
                        ctx_circuit[:, lag_idx]
                    else
                        fut_s = lag_idx - N_ctx

                        if fut_s < block_start_s
                            # Before this block: reset to truth.
                            X_true[:, lag_idx]
                        elseif fut_s < s && all(isfinite.(yhat_closed[:, fut_s]))
                            # Inside this block: use previous prediction.
                            @view(yhat_closed[:, fut_s])
                        else
                            bad_window = true
                            break
                        end
                    end

                    for r in 1:lag_rep
                        s_eff = (layer_base - 1) * lag_rep + r
                        @views inp_s[:, s_eff, 1] .= xlag
                    end
                end

            else
                error("INPUT_MAPPING_MODE must be :qubit or :layer")
            end

            if bad_window
                continue
            end

            R_s = Quantum_Reservoir_Serial_arrangement(inp_s, circuit, B1, nmemory, U, noise_cir)
            Y_s = W * R_s
            @views yhat_closed[:, s] .= Y_s[:, 1]
        end

        block_start_s = block_end_s + 1
    end

    return yhat_closed
end
# ----------------- U and H -----------------

approx_name(method) = method === :plain ? "plain" :
                      method === :Taylor ? "Taylor" :
                      method === :Chebyshev ? "Chebyshev" :
                      method === :Trotter ? "Trotter" :
                      error("Unknown approximation method: $method")

function make_U_plain(H::QubitsOperator, τ::Real)
    A = Matrix(QuantumCircuits.matrix(H))
    U = exp(-im * Float64(τ) * ComplexF64.(A))
    return ComplexF32.(U)
end

function make_U(H, τ)
    if APPROX_METHOD === :plain
        return make_U_plain(H, τ)
    else
        error("APPROX_METHOD=$(APPROX_METHOD) not implemented in this script")
    end
end

function sample_couplings(npairs::Int; scale::Float64=1.0, dist::Symbol=:uniform)
    if dist === :uniform
        return rand(npairs) .* scale
    elseif dist === :normal
        return abs.(randn(npairs)) .* scale
    else
        error("Unknown XYZ_DIST = $dist (use :uniform or :normal)")
    end
end



function normalize_couplings_batch!(Jx::AbstractVector, Jy::AbstractVector, Jz::AbstractVector)
    m = max(maximum(Jx), maximum(Jy), maximum(Jz))
    if m > 0
        Jx ./= m
        Jy ./= m
        Jz ./= m
    end
    return m
end

function Ham_XYZ_nearest_neighbors(
    nqubit::Int,
    Jx::AbstractVector,
    Jy::AbstractVector,
    Jz::AbstractVector;
    periodic::Bool=false,
)
    nneigh = periodic ? nqubit : (nqubit - 1)
    @assert length(Jx) == nneigh && length(Jy) == nneigh && length(Jz) == nneigh

    H = QubitsOperator()

    for i in 1:(nqubit-1)
        j = i + 1
        H += QubitsTerm(i => "X", j => "X", coeff=Jx[i])
        H += QubitsTerm(i => "Y", j => "Y", coeff=Jy[i])
        H += QubitsTerm(i => "Z", j => "Z", coeff=Jz[i])
    end

    if periodic
        H += QubitsTerm(nqubit => "X", 1 => "X", coeff=Jx[end])
        H += QubitsTerm(nqubit => "Y", 1 => "Y", coeff=Jy[end])
        H += QubitsTerm(nqubit => "Z", 1 => "Z", coeff=Jz[end])
    end

    return H
end


function build_measurement_ops(
    input_features::Vector{String},
    nsystem::Int,
    nmemory::Int,
)
    return build_B_ops(
        input_features,
        nsystem,
        nmemory,
    )
end


function get_or_generate_Hs_Us(
    nqubit::Int,
    τ::Float64;
    Batch::Int=100,
    seed::Int=42,
    save_root::AbstractString=PRECOMP_U_DIR,
    xyz_dist::Symbol=XYZ_DIST,
    jx_scale::Float64=XYZ_JX_SCALE,
    jy_scale::Float64=XYZ_JY_SCALE,
    jz_scale::Float64=XYZ_JZ_SCALE,
)
    approx_str = approx_name(APPROX_METHOD)

    isdir(save_root) || mkpath(save_root)

    fname = joinpath(
        save_root,
        @sprintf(
            "HsUs_%s_XYZ_NN_nq%d_tau%g_dist%s_jx%.3f_jy%.3f_jz%.3f_batch%d_seed%d.jld2",
            approx_str,
            nqubit,
            τ,
            String(xyz_dist),
            jx_scale,
            jy_scale,
            jz_scale,
            Batch,
            seed,
        ),
    )

    if isfile(fname)
        @info "Loading precomputed Hs/Us from $fname"
        data = load(fname)
        return data["Hs"], data["Us"]
    end

    @info(
        "Generating new nearest-neighbor XYZ Hs/Us..."
    )

    Random.seed!(seed)

    Hs = Vector{Any}(undef, Batch)
    Us = Vector{Matrix{ComplexF32}}(undef, Batch)

    for fp in 1:Batch
        nneigh = nqubit - 1

        Jx = sample_couplings(
            nneigh;
            scale=jx_scale,
            dist=xyz_dist,
        )

        Jy = sample_couplings(
            nneigh;
            scale=jy_scale,
            dist=xyz_dist,
        )

        Jz = sample_couplings(
            nneigh;
            scale=jz_scale,
            dist=xyz_dist,
        )

        normalize_couplings_batch!(
            Jx,
            Jy,
            Jz,
        )

        H = Ham_XYZ_nearest_neighbors(
            nqubit,
            Jx,
            Jy,
            Jz;
            periodic=false,
        )

        Hs[fp] = H
        Us[fp] = make_U(H, τ)

        if fp % 10 == 0
            @info "Generated $fp / $Batch unitaries..."
        end
    end

    jldsave(
        fname;
        Hs=Hs,
        Us=Us,
        ham_kind=:XYZ_NN,
        seed=seed,
    )

    @info "Saved Hs/Us to $fname"

    return Hs, Us
end

function expected_figure_paths(tag::AbstractString)
    if RUN_CLOSED_LOOP
        closed_tag = closed_loop_mode_name()

        return Dict(
            :rmse => joinpath(FIGURES_DIR, "rmse_vsTime_" * closed_tag * "_MEANBATCH_" * tag * ".png"),
            :bestperfeature => joinpath(FIGURES_DIR, "timeseries_" * closed_tag * "_BESTPERFEATURE_" * tag * ".png"),
            :bestallfeatures => joinpath(FIGURES_DIR, "timeseries_" * closed_tag * "_BESTBATCH_ALLFEATURES_" * tag * "_together.png"),
            :ctxplusfut => joinpath(FIGURES_DIR, "timeseries_" * closed_tag * "_BESTPERFEATURE_CTXPLUSFUT_" * tag * ".png"),
        )
    else
        return Dict(
            :rmse => joinpath(FIGURES_DIR, "rmse_vsTime_OPEN_MEANBATCH_" * tag * ".png"),
            :bestperfeature => joinpath(FIGURES_DIR, "timeseries_OPEN_BESTPERFEATURE_" * tag * ".png"),
            :bestallfeatures => joinpath(FIGURES_DIR, "timeseries_OPEN_BESTBATCH_ALLFEATURES_" * tag * "_together.png"),
            :ctxplusfut => joinpath(FIGURES_DIR, "timeseries_OPEN_BESTPERFEATURE_CTXPLUSFUT_" * tag * ".png"),
        )
    end
end

function missing_figure_keys(tag::AbstractString)
    MAKE_PREDICTION_FIGURES ||
        return Symbol[]

    paths = expected_figure_paths(tag)

    return [
        k for (k, p) in paths
        if !isfile(p)
    ]
end

function per_feature_rmse_stats(pred_all, fut_plot)
    F = size(fut_plot, 1)
    B = length(pred_all)

    rmse_per_batch = fill(NaN, F, B)
    avg_rmse = fill(NaN, F)
    best_fp = fill(0, F)
    best_score = fill(NaN, F)

    for f in 1:F
        best_val = Inf
        best_b = 0
        vals = Float64[]

        for fp in 1:B
            yhat = pred_all[fp][f, :]
            d = yhat .- fut_plot[f, :]
            m = isfinite.(d)

            r = any(m) ? sqrt(mean(abs2, d[m])) : NaN
            rmse_per_batch[f, fp] = r

            if isfinite(r)
                push!(vals, r)
                if r < best_val
                    best_val = r
                    best_b = fp
                end
            end
        end

        avg_rmse[f] = isempty(vals) ? NaN : mean(vals)
        best_fp[f] = best_b
        best_score[f] = isfinite(best_val) ? best_val : NaN
    end

    return rmse_per_batch, avg_rmse, best_fp, best_score
end

function overall_rmse_stats(pred_all, fut_plot)
    B = length(pred_all)
    F = size(fut_plot, 1)

    rmse_overall_per_batch = fill(NaN, B)

    for fp in 1:B
        errs_sq = Float64[]

        for f in 1:F
            yhat = pred_all[fp][f, :]
            d = yhat .- fut_plot[f, :]
            m = isfinite.(d)

            if any(m)
                append!(errs_sq, abs2.(d[m]))
            end
        end

        rmse_overall_per_batch[fp] = isempty(errs_sq) ? NaN : sqrt(mean(errs_sq))
    end

    good = isfinite.(rmse_overall_per_batch)
    avg_all = any(good) ? mean(rmse_overall_per_batch[good]) : NaN
    best_fp = any(good) ? argmin(rmse_overall_per_batch) : 0
    best_score = best_fp == 0 ? NaN : rmse_overall_per_batch[best_fp]

    return rmse_overall_per_batch, avg_all, best_fp, best_score
end

function regenerate_missing_figures_from_jld!(
    jld_path::AbstractString,
    tag::AbstractString,
    meta_str_common::AbstractString,
)
    MAKE_PREDICTION_FIGURES ||
        return nothing

    fig_paths = expected_figure_paths(tag)
    missing = missing_figure_keys(tag)

    if isempty(missing)
        @info "JLD exists and all figures already exist for $(basename(jld_path))"
        return
    end

    @info "JLD exists but missing figures for $(basename(jld_path)): $(missing)"
    data = load(jld_path)

    input_features = String.(data["INPUT_FEATURES"])
    feature_size = length(input_features)

    fut_plot = data["fut_plot"]
    t_axis = data["t_axis"]
    N_fut = size(fut_plot, 2)

    ahead = Int(data["ahead"])

    c_open = :orange
    c_closed = :green
    c_true = :blue
    legend_fs = 9
    legend_bg = RGBA(1, 1, 1, 0.55)
    t_min, t_max = PLOT_T_RANGE
    mask = (t_axis .>= t_min) .& (t_axis .<= t_max)

    is_closed = Bool(data["RUN_CLOSED_LOOP"])

    if is_closed
        pred_all = data["closed_pred_zscore_all"]
        rmse_vs_time = data["closed_rmse_vs_time"]

        rmse_per_batch, avg_rmse, best_fp_per_feature, best_score =
            per_feature_rmse_stats(pred_all, fut_plot)

        rmse_overall_per_batch, rmse_all_features_all_batches, best_fp_together, best_score_together =
            overall_rmse_stats(pred_all, fut_plot)
    else
        pred_all = data["open_pred_zscore_all"]
        rmse_vs_time = data["open_rmse_vs_time"]

        rmse_per_batch, avg_rmse, best_fp_per_feature, best_score =
            per_feature_rmse_stats(pred_all, fut_plot)

        rmse_overall_per_batch, rmse_all_features_all_batches, best_fp_together, best_score_together =
            overall_rmse_stats(pred_all, fut_plot)
    end

    if :rmse in missing
        p_rmse = plot(
            layout=(feature_size + 1, 1),
            size=(900, 250 * (feature_size + 1)),
            left_margin=20mm,
            right_margin=20mm,
            titlefont=14,
            guidefont=12,
            tickfont=11,
        )

        for f in 1:feature_size
            plot!(p_rmse[f], t_axis[mask], @view(rmse_vs_time[f, mask]);
                lw=2,
                seriescolor=is_closed ? c_closed : c_open,
                marker=:circle,
                markerstrokecolor=:auto,
                markersize=2,
                label=is_closed ? "Closed (mean over batches)" : "Open (mean over batches)",
                ylabel=input_features[f] * " RMSE (norm)")
            title!(p_rmse[f], @sprintf("%s  (%s)", input_features[f], is_closed ? "CLOSED" : "OPEN"))
            plot!(p_rmse[f]; legend=:topright)
        end

        xlabel!(p_rmse[feature_size], time_axis_label())

        idx_meta = feature_size + 1
        plot!(p_rmse[idx_meta], [0, 1], [0, 1];
            legend=false,
            framestyle=:none,
            xticks=false,
            yticks=false,
            linealpha=0)

        annotate!(p_rmse[idx_meta], 0.5, 0.5,
            text(meta_str_common * "\n" * @sprintf("%s, Horizon=%d", is_closed ? "CLOSED" : "OPEN", ahead), 14, :center))

        # savefig(p_rmse, fig_paths[:rmse])
        @info "Recovered RMSE figure: $(fig_paths[:rmse])"
    end

    if :bestperfeature in missing
        p_ts = plot(
            layout=(feature_size + 1, 1),
            size=(900, 250 * (feature_size + 1)),
            left_margin=20mm,
            right_margin=20mm,
            titlefont=14,
            guidefont=12,
            tickfont=11,
        )

        for f in 1:feature_size
            y_true_f = @view fut_plot[f, :]
            y_pred_f = fill(NaN, N_fut)

            fp = best_fp_per_feature[f]
            if fp != 0
                y_pred_f .= @view pred_all[fp][f, :]
            end

            plot!(p_ts[f], t_axis[mask], y_true_f[mask];
                lw=2,
                seriescolor=c_true,
                marker=:circle,
                markerstrokecolor=:auto,
                markersize=2,
                label="True")

            plot!(p_ts[f], t_axis[mask], y_pred_f[mask];
                lw=2,
                seriescolor=is_closed ? c_closed : c_open,
                marker=:circle,
                markerstrokecolor=:auto,
                markersize=2,
                label = fp == 0 ? (is_closed ? "Closed" : "Open") :
                    @sprintf(
                        "%s b%02d, best %.1e, mean %.3e",
                        is_closed ? "Closed" : "Open",
                        fp,
                        best_score[f],
                        avg_rmse[f],
                    ))

            ylabel!(p_ts[f], input_features[f] * " (norm)")
            title!(p_ts[f], @sprintf("%s  (%s)", input_features[f], is_closed ? "CLOSED" : "OPEN"))

            plot!(p_ts[f];
                legend=:bottomright,
                legendfontsize=legend_fs,
                background_color_legend=legend_bg)
        end

        xlabel!(p_ts[feature_size], time_axis_label())

        idx_meta = feature_size + 1
        plot!(p_ts[idx_meta], [0, 1], [0, 1];
            legend=false,
            framestyle=:none,
            xticks=false,
            yticks=false,
            linealpha=0)

        annotate!(p_ts[idx_meta], 0.5, 0.5,
            text(meta_str_common * "\n" * @sprintf("%s-loop horizon=%d", is_closed ? "CLOSED" : "OPEN", ahead), 14, :center))

        # savefig(p_ts, fig_paths[:bestperfeature])
        @info "Recovered BESTPERFEATURE figure: $(fig_paths[:bestperfeature])"
    end

    if :bestallfeatures in missing
        p_together = plot(
            layout=(feature_size + 1, 1),
            size=(900, 250 * (feature_size + 1)),
            left_margin=20mm,
            right_margin=20mm,
            titlefont=14,
            guidefont=12,
            tickfont=11,
        )

        fpT = best_fp_together

        for f in 1:feature_size
            y_true_f = @view fut_plot[f, :]
            y_pred_f = fill(NaN, N_fut)

            if fpT != 0
                y_pred_f .= @view pred_all[fpT][f, :]
            end

            plot!(p_together[f], t_axis[mask], y_true_f[mask];
                lw=2,
                seriescolor=c_true,
                marker=:circle,
                markerstrokecolor=:auto,
                markersize=2,
                label="True")

            plot!(p_together[f], t_axis[mask], y_pred_f[mask];
                lw=2,
                seriescolor=is_closed ? c_closed : c_open,
                marker=:circle,
                markerstrokecolor=:auto,
                markersize=2,
                label=fpT == 0 ? "$(is_closed ? "Closed" : "Open") (no batch)" :
                    @sprintf(
                        "%s b%02d (together shown)\nRMSE(all features all batches) %.2e",
                        is_closed ? "Closed" : "Open",
                        fpT,
                        rmse_all_features_all_batches,
                    ))

            ylabel!(p_together[f], input_features[f] * " (norm)")
            title!(p_together[f], @sprintf("%s  (%s, Horizon=%d)", input_features[f], is_closed ? "CLOSED" : "OPEN", ahead))
            plot!(p_together[f]; legend=:outerright, legendfontsize=legend_fs)
        end

        xlabel!(p_together[feature_size], time_axis_label())

        idx_meta = feature_size + 1
        plot!(p_together[idx_meta], [0, 1], [0, 1];
            legend=false,
            framestyle=:none,
            xticks=false,
            yticks=false,
            linealpha=0)

        annotate!(p_together[idx_meta], 0.5, 0.5,
            text(meta_str_common * "\n" *
                @sprintf(
                    "%s-loop horizon=%d\nBEST BATCH OVER ALL FEATURES: b%02d  RMSE(all)=%.2e",
                    is_closed ? "CLOSED" : "OPEN",
                    ahead,
                    fpT,
                    best_score_together,
                ),
                14,
                :center))

        # savefig(p_together, fig_paths[:bestallfeatures])
        @info "Recovered BESTBATCH_ALLFEATURES figure: $(fig_paths[:bestallfeatures])"
    end

    if :ctxplusfut in missing
        # For MAPLIKE=true, exact CTX+FUT recovery needs ctx_norm_raw, fut_norm_raw,
        # t_ctx_raw, and t_fut_raw saved in the JLD. For raw mode, this simple
        # recovery is enough.
        if MAPLIKE
            @warn "Cannot fully recover CTXPLUSFUT maplike figure from this JLD unless ctx_norm_raw, fut_norm_raw, t_ctx_raw, and t_fut_raw are saved."
        else
            p_ctx = plot(
                layout=(feature_size + 1, 1),
                size=(900, 250 * (feature_size + 1)),
                left_margin=20mm,
                right_margin=20mm,
                titlefont=14,
                guidefont=12,
                tickfont=11,
            )

            for f in 1:feature_size
                fp = best_fp_per_feature[f]

                y_true_f = @view fut_plot[f, :]
                y_pred_f = fill(NaN, N_fut)

                if fp != 0
                    y_pred_f .= @view pred_all[fp][f, :]
                end

                plot!(p_ctx[f], t_axis, y_true_f;
                    lw=2,
                    seriescolor=c_true,
                    label="True")

                plot!(p_ctx[f], t_axis, y_pred_f;
                    lw=2,
                    seriescolor=is_closed ? c_closed : c_open,
                    label=is_closed ? "Closed" : "Open")

                ylabel!(p_ctx[f], input_features[f] * " (norm)")
                title!(p_ctx[f], @sprintf("%s  (%s+CTX view, Horizon=%d)", input_features[f], is_closed ? "CLOSED" : "OPEN", ahead))
                plot!(p_ctx[f]; legend=false)
            end

            xlabel!(p_ctx[feature_size], time_axis_label())

            idx_meta = feature_size + 1
            plot!(p_ctx[idx_meta], [0, 1], [0, 1];
                legend=false,
                framestyle=:none,
                xticks=false,
                yticks=false,
                linealpha=0)

            annotate!(p_ctx[idx_meta], 0.5, 0.5,
                text(meta_str_common * "\n" * @sprintf("%s-loop horizon=%d (CTX+FUT view)", is_closed ? "CLOSED" : "OPEN", ahead), 14, :center))

            # savefig(p_ctx, fig_paths[:ctxplusfut])
            @info "Recovered CTXPLUSFUT figure: $(fig_paths[:ctxplusfut])"
        end
    end
end

# ============================================================
# DECOHERENCE ROBUSTNESS SWEEP
#
# Blue:
#   γ_train = 0
#   γ_test  = γ
#
# Red:
#   γ_train = γ
#   γ_test  = γ
# ============================================================

function finite_rmse(
    prediction::AbstractArray,
    target::AbstractArray,
)
    size(prediction) == size(target) ||
        error(
            "Prediction and target sizes differ: " *
            "$(size(prediction)) versus $(size(target))."
        )

    difference =
        Float64.(prediction) .-
        Float64.(target)

    valid = isfinite.(difference)

    return any(valid) ?
        sqrt(mean(abs2, difference[valid])) :
        NaN
end


function finite_mean_std(values)
    finite_values = Float64[
        value for value in vec(values)
        if isfinite(value)
    ]

    isempty(finite_values) &&
        return NaN, NaN

    return (
        mean(finite_values),
        std(finite_values; corrected=false),
    )
end

function load_existing_decoherence_rows!(
    gamma_values::Vector{Float64},
    target_rmse_trained_ideal::Matrix{Float64},
    target_rmse_trained_with_decoherence::Matrix{Float64},
    base_tag::String,
    τ::Float64,
)
    completed_gamma =
        falses(length(gamma_values))

    isdir(RESULTS_DIR) ||
        return completed_gamma

    # Accept:
    #
    #   base_tag_resume.jld2
    #
    # and previous completed files such as:
    #
    #   base_tag_g0.000-0.050_ngamma2.jld2
    candidate_paths = String[]

    for filename in readdir(RESULTS_DIR)
        is_candidate =
            filename == base_tag * "_resume.jld2" ||
            (
                startswith(
                    filename,
                    base_tag * "_g",
                ) &&
                endswith(
                    filename,
                    ".jld2",
                )
            )

        if is_candidate
            push!(
                candidate_paths,
                joinpath(
                    RESULTS_DIR,
                    filename,
                ),
            )
        end
    end

    # Load older files first. A newer valid result can replace
    # an older result for the same γ.
    sort!(
        candidate_paths;
        by=path -> stat(path).mtime,
    )

    for path in candidate_paths
        data = try
            JLD2.load(path)
        catch err
            @warn(
                "Could not load previous decoherence file; ignoring it.",
                path=path,
                exception=(
                    err,
                    catch_backtrace(),
                ),
            )

            continue
        end

        required_keys = (
            "gamma",
            "target_rmse_trained_ideal",
            "target_rmse_trained_with_decoherence",
        )

        all(key -> haskey(data, key), required_keys) || continue

        probability_key =
            haskey(data, "channel_probability") ?
            "channel_probability" :
            "phase_damping_probability"

        haskey(data, probability_key) || continue


        # Do not reuse data obtained with another batch count.
        if haskey(data, "number_of_batches")
            Int(data["number_of_batches"]) ==
            N_BATCHES || continue
        end

        # Do not reuse data obtained with another ridge value.
        if haskey(data, "lambda")
            isapprox(
                Float64(data["lambda"]),
                Float64(λ);
                atol=0.0,
                rtol=0.0,
            ) || continue
        end

        old_gamma =
            Float64.(
                vec(data["gamma"])
            )

        old_ideal =
            Matrix{Float64}(
                data[
                    "target_rmse_trained_ideal"
                ]
            )

        old_decoherent =
            Matrix{Float64}(
                data[
                    "target_rmse_trained_with_decoherence"
                ]
            )

        expected_size = (
            length(old_gamma),
            N_BATCHES,
        )

        size(old_ideal) == expected_size ||
            continue

        size(old_decoherent) == expected_size ||
            continue

        # Reject old files made with the previous, incorrect
        # conversion between γ and PhaseDamping p.
        old_probability =
            Float64.(
                vec(
                    data[
                        probability_key
                    ]
                )
            )

        expected_probability =
            [
                decoherence_probability(τ, γ)
                for γ in old_gamma
            ]

        length(old_probability) ==
        length(expected_probability) ||
            continue

        probability_matches =
            all(
                index ->
                    isapprox(
                        old_probability[index],
                        expected_probability[index];
                        atol=1e-10,
                        rtol=1e-10,
                    ),
                eachindex(old_probability),
            )

        probability_matches ||
            continue

        for (
            requested_index,
            requested_gamma,
        ) in enumerate(gamma_values)

            old_index =
                findfirst(
                    old_value ->
                        isapprox(
                            old_value,
                            requested_gamma;
                            atol=
                                DECOHERENCE_GAMMA_ATOL,
                            rtol=0.0,
                        ),
                    old_gamma,
                )

            old_index === nothing &&
                continue

            old_ideal_row =
                @view old_ideal[
                    old_index,
                    :,
                ]

            old_decoherent_row =
                @view old_decoherent[
                    old_index,
                    :,
                ]

            # Reuse the point only when all batches were
            # successfully completed.
            row_is_complete =
                all(
                    isfinite,
                    old_ideal_row,
                ) &&
                all(
                    isfinite,
                    old_decoherent_row,
                )

            row_is_complete ||
                continue

            @views target_rmse_trained_ideal[
                requested_index,
                :,
            ] .= old_ideal_row

            @views target_rmse_trained_with_decoherence[
                requested_index,
                :,
            ] .= old_decoherent_row

            completed_gamma[
                requested_index
            ] = true

            println(
                "Loaded γ=$requested_gamma from: " *
                basename(path)
            )
        end
    end

    return completed_gamma
end

function decoherence_profile_mode_name()
    if !USE_GAUSSIAN_QUBIT_RATES
        return "homogeneous"

    elseif GAMMA_STD_MODE === :fractional
        return "gaussian_qubit_fractional_std"

    elseif GAMMA_STD_MODE === :fixed
        return "gaussian_qubit_fixed_std"

    else
        error(
            "Invalid GAMMA_STD_MODE=$GAMMA_STD_MODE. " *
            "Use :fractional or :fixed."
        )
    end
end

function build_rate_probability_tensors(
    gamma_values::Vector{Float64},
    nqubit::Int,
    τ::Real;
    draw_index::Int,
)
    number_of_gamma = length(gamma_values)

    rate_tensor =
        Array{Float64}(undef, number_of_gamma, N_BATCHES, nqubit)

    probability_tensor =
        Array{Float64}(undef, number_of_gamma, N_BATCHES, nqubit)

    zero_rate_tensor =
        zeros(Float64, number_of_gamma, N_BATCHES, nqubit)

    zero_probability_tensor =
        zeros(Float64, number_of_gamma, N_BATCHES, nqubit)

    rate_mean_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    rate_std_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    rate_min_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    rate_max_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    probability_mean_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    probability_std_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    probability_min_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    probability_max_per_gamma_batch =
        Array{Float64}(undef, number_of_gamma, N_BATCHES)

    for gamma_index in eachindex(gamma_values)
        for fp in 1:N_BATCHES
            rates = sampled_gamma_vector(
                gamma_values[gamma_index],
                nqubit;
                gamma_index=gamma_index,
                batch_index=fp,
                draw_index=draw_index,
            )

            @views rate_tensor[gamma_index, fp, :] .= rates

            for q in 1:nqubit
                probability_tensor[gamma_index, fp, q] =
                    decoherence_probability(τ, rates[q])
            end

            r_view = @view rate_tensor[gamma_index, fp, :]
            p_view = @view probability_tensor[gamma_index, fp, :]

            rate_mean_per_gamma_batch[gamma_index, fp] =
                mean(r_view)

            rate_std_per_gamma_batch[gamma_index, fp] =
                std(r_view; corrected=false)

            rate_min_per_gamma_batch[gamma_index, fp] =
                minimum(r_view)

            rate_max_per_gamma_batch[gamma_index, fp] =
                maximum(r_view)

            probability_mean_per_gamma_batch[gamma_index, fp] =
                mean(p_view)

            probability_std_per_gamma_batch[gamma_index, fp] =
                std(p_view; corrected=false)

            probability_min_per_gamma_batch[gamma_index, fp] =
                minimum(p_view)

            probability_max_per_gamma_batch[gamma_index, fp] =
                maximum(p_view)
        end
    end

    return (
        rate_tensor=rate_tensor,
        probability_tensor=probability_tensor,
        zero_rate_tensor=zero_rate_tensor,
        zero_probability_tensor=zero_probability_tensor,

        rate_mean_per_gamma_batch=rate_mean_per_gamma_batch,
        rate_std_per_gamma_batch=rate_std_per_gamma_batch,
        rate_min_per_gamma_batch=rate_min_per_gamma_batch,
        rate_max_per_gamma_batch=rate_max_per_gamma_batch,

        probability_mean_per_gamma_batch=probability_mean_per_gamma_batch,
        probability_std_per_gamma_batch=probability_std_per_gamma_batch,
        probability_min_per_gamma_batch=probability_min_per_gamma_batch,
        probability_max_per_gamma_batch=probability_max_per_gamma_batch,
    )
end

function save_decoherence_resume!(
    resume_path::String;
    gamma_values::Vector{Float64},
    target_rmse_trained_ideal::Matrix{Float64},
    target_rmse_trained_with_decoherence::Matrix{Float64},
    base_tag::String,
    config_name::String,
    target_feature::String,
    input_features::Vector{String},
    ntimesteps::Int,
    nmemory::Int,
    nsystem::Int,
    nqubit::Int,
    nlayers::Int,
    ds::Int,
    ahead::Int,
    τ::Float64,
)
    completed_gamma = [
        all(
            isfinite,
            @view(
                target_rmse_trained_ideal[
                    gamma_index,
                    :,
                ]
            ),
        ) &&
        all(
            isfinite,
            @view(
                target_rmse_trained_with_decoherence[
                    gamma_index,
                    :,
                ]
            ),
        )
        for gamma_index in eachindex(gamma_values)
    ]

    channel_probability =
        [
            decoherence_probability(τ, γ)
            for γ in gamma_values
        ]
    train_rate_probability_pack =
        build_rate_probability_tensors(
            gamma_values,
            nqubit,
            τ;
            draw_index=1,
        )

    test_rate_probability_pack =
        build_rate_probability_tensors(
            gamma_values,
            nqubit,
            τ;
            draw_index=2,
        )
    temporary_path =
        resume_path * ".tmp.jld2"

    isfile(temporary_path) &&
        rm(
            temporary_path;
            force=true,
        )

    jldsave(
        temporary_path;

        resume_version=1,
        base_tag=base_tag,

        DATA_MODE=String(DATA_MODE),
        configuration=config_name,
        target_feature=target_feature,
        INPUT_FEATURES=input_features,

        INPUT_MAPPING_MODE=
            String(INPUT_MAPPING_MODE),

        NTIMESTEPS=ntimesteps,
        N_MEMORY=nmemory,
        NSYSTEM=nsystem,
        nqubit=nqubit,
        N_LAYERS=nlayers,
        DS=ds,
        ahead=ahead,

        HAM_NAME=HAM_NAME,
        hamiltonian_kind=
            String(HAMILTONIAN_KIND),

        EVOLUTION_TIME_MODE=
            String(EVOLUTION_TIME_MODE),

        τ=Float64(τ),
        lambda=Float64(λ),
        number_of_batches=N_BATCHES,
        normalization_mode=
            String(NORM_MODE),

        noise_model=decoherence_model_name(),

        decoherence_calculation_mode=
            String(DECOHERENCE_CALCULATION_MODE),

        decoherence_convention=
            DECOHERENCE_MODEL === :dephasing ?
            "p=1-exp(-4*gamma*tau)" :
            "p=1-exp(-kappa*tau)",

        # Mean-rate list.
        # For Gaussian qubit rates, this is the requested mean rate.
        gamma=gamma_values,
        mean_rate=gamma_values,

        rate_symbol=
            DECOHERENCE_MODEL === :dephasing ?
            "gamma" :
            "kappa",

        gamma_is_mean_rate=
            USE_GAUSSIAN_QUBIT_RATES,

        gamma_profile_mode=
            decoherence_profile_mode_name(),

        gamma_profile_tag=
            gamma_profile_tag(),

        use_gaussian_qubit_rates=
            USE_GAUSSIAN_QUBIT_RATES,

        gamma_std_mode=
            String(GAMMA_STD_MODE),

        gamma_std_fraction=
            Float64(GAMMA_STD_FRACTION),

        gamma_fixed_std=
            Float64(GAMMA_FIXED_STD),

        gamma_std_used=[
            gamma_standard_deviation(γ)
            for γ in gamma_values
        ],

        gamma_profile_seed=
            Int(GAMMA_PROFILE_SEED),

        rate_vector_shape=
            "gamma_index × batch_index × qubit_index",

        # Probability at the listed mean rate.
        # Kept for compatibility with older loading code.
        channel_probability=
            channel_probability,

        channel_probability_at_mean_rate=
            channel_probability,

        # Keep this only for backward compatibility with older scripts.
        phase_damping_probability=
            channel_probability,

        # Backward-compatible scalar fields.
        blue_training_rate=0.0,
        blue_test_rate=
            gamma_values,

        red_training_rate=
            gamma_values,

        red_test_rate=
            gamma_values,

        # Actual qubit-dependent rates used in the calculation.
        blue_training_rate_vectors=
            train_rate_probability_pack.zero_rate_tensor,

        blue_test_rate_vectors=
            test_rate_probability_pack.rate_tensor,

        red_training_rate_vectors=
            train_rate_probability_pack.rate_tensor,

        red_test_rate_vectors=
            test_rate_probability_pack.rate_tensor,

        # Actual qubit-dependent channel probabilities.
        blue_training_channel_probability_vectors=
            train_rate_probability_pack.zero_probability_tensor,

        blue_test_channel_probability_vectors=
            test_rate_probability_pack.probability_tensor,

        red_training_channel_probability_vectors=
            train_rate_probability_pack.probability_tensor,

        red_test_channel_probability_vectors=
            test_rate_probability_pack.probability_tensor,

        # Useful summaries over qubits for the test-noise draw.
        qubit_rate_mean_per_gamma_batch=
            test_rate_probability_pack.rate_mean_per_gamma_batch,

        qubit_rate_std_per_gamma_batch=
            test_rate_probability_pack.rate_std_per_gamma_batch,

        qubit_rate_min_per_gamma_batch=
            test_rate_probability_pack.rate_min_per_gamma_batch,

        qubit_rate_max_per_gamma_batch=
            test_rate_probability_pack.rate_max_per_gamma_batch,

        qubit_probability_mean_per_gamma_batch=
            test_rate_probability_pack.probability_mean_per_gamma_batch,

        qubit_probability_std_per_gamma_batch=
            test_rate_probability_pack.probability_std_per_gamma_batch,

        qubit_probability_min_per_gamma_batch=
            test_rate_probability_pack.probability_min_per_gamma_batch,

        qubit_probability_max_per_gamma_batch=
            test_rate_probability_pack.probability_max_per_gamma_batch,

        completed_gamma=
            completed_gamma,

        target_rmse_trained_ideal=
            target_rmse_trained_ideal,

        target_rmse_trained_with_decoherence=
            target_rmse_trained_with_decoherence,
    )

    mv(
        temporary_path,
        resume_path;
        force=true,
    )

    return nothing
end

function run_decoherence_sweep(
    ;
    config_name::String,
    target_feature::String,
    input_features::Vector{String},
    ntimesteps::Int,
    nmemory::Int,
    ds::Int,
    nlayers::Int=1,
    ahead::Int=0,
)
    # ========================================================
    # Validate the selected experiment
    # ========================================================

    RUN_CLOSED_LOOP == false ||
        error(
            "The decoherence sweep supports only open-loop " *
            "teacher-forced prediction. Set RUN_CLOSED_LOOP=false."
        )

    RUN_BLOCKWISE_CLOSED_LOOP == false ||
        error(
            "Set RUN_BLOCKWISE_CLOSED_LOOP=false."
        )

    INPUT_MAPPING_MODE === :qubit ||
        error(
            "The selected I/M configurations were obtained for " *
            "INPUT_MAPPING_MODE=:qubit."
        )

    DATA_MODE in (:theory, :experiment) ||
        error(
            "The selected u1/u2/iB/iC configurations support only " *
            "DATA_MODE=:theory or DATA_MODE=:experiment."
        )

    MAPLIKE == false ||
        error(
            "The selected configurations were obtained for raw " *
            "time-series prediction. Set MAPLIKE=false."
        )

    REPEAT_FEATURES == 1 ||
        error(
            "The selected configurations require REPEAT_FEATURES=1."
        )

    LAG_REP == 1 ||
        error(
            "The selected configurations require LAG_REP=1."
        )

    nlayers == 1 ||
        error(
            "Qubit mapping requires nlayers=1 for these configurations."
        )

    ahead == 0 ||
        error(
            "The selected configurations require ahead=0."
        )

    isempty(GAMMA_LIST) &&
        error("GAMMA_LIST cannot be empty.")

    all(
        γ -> isfinite(γ) && γ >= 0,
        GAMMA_LIST,
    ) ||
        error(
            "Every value in GAMMA_LIST must be finite and nonnegative."
        )

    # Confirm that the supplied configuration is one of the
    # explicitly selected entries in DECOHERENCE_JOBS.
    selected_job_index = findfirst(
        job ->
            job.data_mode === DATA_MODE &&
            job.config == config_name &&
            job.target == target_feature,
        DECOHERENCE_JOBS,
    )

    selected_job_index === nothing &&
        error(
            "No selected decoherence job exists for " *
            "DATA_MODE=$DATA_MODE, configuration=$config_name, " *
            "and target=$target_feature."
        )
    expected_job =
        DECOHERENCE_JOBS[selected_job_index]

    input_features == expected_job.features ||
        error(
            "Incorrect input features for $config_name, " *
            "target=$target_feature.\n" *
            "Expected: $(expected_job.features)\n" *
            "Received: $input_features"
        )

    ntimesteps == expected_job.ntimesteps ||
        error(
            "Incorrect ntimesteps for $config_name, " *
            "target=$target_feature. " *
            "Expected $(expected_job.ntimesteps), " *
            "received $ntimesteps."
        )

    nmemory == expected_job.nmemory ||
        error(
            "Incorrect nmemory for $config_name, " *
            "target=$target_feature. " *
            "Expected $(expected_job.nmemory), " *
            "received $nmemory."
        )

    ds == expected_job.ds ||
        error(
            "Incorrect ds for $config_name, " *
            "target=$target_feature. " *
            "Expected $(expected_job.ds), received $ds."
        )

    target_feature in input_features ||
        error(
            "Target $target_feature is not present in " *
            "input_features=$input_features."
        )

    ntimesteps > 0 ||
        error("ntimesteps must be positive.")

    nmemory >= 0 ||
        error("nmemory must be nonnegative.")

    ds > 0 ||
        error("ds must be positive.")

    # ========================================================
    # Reservoir size
    #
    # :qubit mapping:
    # nsystem = features × timesteps × lag repetitions
    # ========================================================

    nsystem =
        length(input_features) *
        ntimesteps *
        LAG_REP *
        REPEAT_FEATURES

    nqubit =
        nsystem + nmemory

    # nqubit == 11 ||
    #     error(
    #         "Every selected configuration must contain 11 total " *
    #         "qubits, but this job produced nqubit=$nqubit.\n" *
    #         "nsystem=$nsystem, nmemory=$nmemory."
    #     )

    MAX_TOTAL_QUBITS >= nqubit ||
        error(
            "MAX_TOTAL_QUBITS=$MAX_TOTAL_QUBITS is smaller than " *
            "the required nqubit=$nqubit."
        )

    τ = evolution_time(
        nqubit
    )

    println()
    println("==============================================")
    println("DECOHERENCE ROBUSTNESS SWEEP")
    println("DATA_MODE       = $DATA_MODE")
    println("configuration   = $config_name")
    println("target          = $target_feature")
    println("input features  = $input_features")
    println("ntimesteps (I)  = $ntimesteps")
    println("memory (M)      = $nmemory")
    println("ds              = $ds")
    println("nlayers         = $nlayers")
    println("ahead           = $ahead")
    println("nsystem         = $nsystem")
    println("nqubit          = $nqubit")
    println("τ               = $τ")
    println("batches         = $N_BATCHES")
    println("calculation     = $DECOHERENCE_CALCULATION_MODE")
    println("==============================================")

    # ========================================================
    # Load the job-specific feature set
    # ========================================================

    data_file = data_file_for(
        BOARD_ID,
        TARGET_RUN_ID,
    )

    isfile(data_file) ||
        error(
            "Data file not found: $data_file"
        )

    data_pack = load_and_prepare_data(
        data_file;
        norm=NORM_MODE,
        input_features=input_features,
        maplike=false,
        maplike_which=MAPLIKE_WHICH,
        maplike_target=MAPLIKE_TARGET,
        apply_lowpass=DATA_MODE === :experiment,
        default_cutoff_Hz=DEFAULT_CUTOFF_HZ,
        default_filter_order=DEFAULT_FILTER_ORDER,
        u2_cutoff_Hz=U2_CUTOFF_HZ,
        u2_filter_order=U2_FILTER_ORDER,
    )

    # ========================================================
    # Prepare reservoir input, targets and evaluation data
    # ========================================================

    ctx_circuit = repeat_features_matrix(
        data_pack.ctx_used,
        REPEAT_FEATURES,
    )

    fut_circuit = repeat_features_matrix(
        data_pack.fut_used,
        REPEAT_FEATURES,
    )

    ctx_target = repeat_features_matrix(
        data_pack.ctx_target,
        REPEAT_FEATURES,
    )

    fut_target = repeat_features_matrix(
        data_pack.fut_target,
        REPEAT_FEATURES,
    )

    true_fut_matrix = repeat_features_matrix(
        data_pack.fut_zscore_eval,
        REPEAT_FEATURES,
    )

    μang_rep = repeat(
        data_pack.μang,
        REPEAT_FEATURES,
    )

    σang_rep = repeat(
        data_pack.σang,
        REPEAT_FEATURES,
    )

    feature_names = repeat(
        input_features,
        REPEAT_FEATURES,
    )

    size(ctx_circuit, 1) == length(feature_names) ||
        error(
            "Context feature-count mismatch: " *
            "$(size(ctx_circuit, 1)) rows versus " *
            "$(length(feature_names)) feature names."
        )

    size(fut_circuit, 1) == length(feature_names) ||
        error(
            "Future feature-count mismatch."
        )

    size(true_fut_matrix, 1) == length(feature_names) ||
        error(
            "Evaluation target feature-count mismatch."
        )

    target_index = findfirst(
        ==(target_feature),
        feature_names,
    )

    target_index === nothing &&
        error(
            "Target $target_feature was not found in " *
            "feature_names=$feature_names."
        )

    # ========================================================
    # Measurement operators and reservoir unitaries
    # ========================================================

    B1 = build_measurement_ops(
        input_features,
        nsystem,
        nmemory,
    )

    _, Us = get_or_generate_Hs_Us(
        nqubit,
        Float64(τ);
        Batch=N_BATCHES,
        seed=42,
    )

    gamma_values =
        Float64.(GAMMA_LIST)

    length(unique(gamma_values)) ==
    length(gamma_values) ||
        error(
            "GAMMA_LIST contains duplicate values: " *
            "$GAMMA_LIST"
        )

    number_of_gamma =
        length(gamma_values)

    # Build a stable filename that does not depend on the
    # beginning, end, or number of γ values.
    source_tag = if DATA_MODE === :theory
        "theory"
    else
        @sprintf(
            "experiment_Vs%.2f_board%d",
            TARGET_VS,
            BOARD_ID,
        )
    end

    feature_tag =
        join(
            input_features,
            "-",
        )

    base_tag = replace(
        @sprintf(
            "decoherence_%s_%s_%s_%s_config_%s_target_%s_features_%s_ham_%s_nq%d_nsys%d_mem%d_mode%s_nt%d_layers%d_ds%d_ahead%d_tau%g_batch%d_norm%s",
            source_tag,
            noise_short_tag(),
            calculation_short_tag(),
            profile_short_tag(),
            config_name,
            target_feature,
            feature_tag,
            HAM_NAME,
            nqubit,
            nsystem,
            nmemory,
            String(INPUT_MAPPING_MODE),
            ntimesteps,
            nlayers,
            ds,
            ahead,
            τ,
            N_BATCHES,
            String(NORM_MODE),
        ),
        r"[^\w\-.]+" => "_",
    )

    output_tag = @sprintf(
        "%s_g%.3f-%.3f_ngamma%d",
        base_tag,
        first(gamma_values),
        last(gamma_values),
        number_of_gamma,
    )



    isdir(RESULTS_DIR) ||
        mkpath(RESULTS_DIR)

    isdir(FIGURES_DIR) ||
        mkpath(FIGURES_DIR)

    resume_path = joinpath(
        RESULTS_DIR,
        base_tag * "_resume.jld2",
    )

    # Dimensions:
    #
    # gamma × reservoir batch
    target_rmse_trained_ideal = fill(
        NaN,
        number_of_gamma,
        N_BATCHES,
    )

    target_rmse_trained_with_decoherence = fill(
        NaN,
        number_of_gamma,
        N_BATCHES,
    )

    completed_gamma =
        if RESUME_DECOHERENCE
            load_existing_decoherence_rows!(
                gamma_values,
                target_rmse_trained_ideal,
                target_rmse_trained_with_decoherence,
                base_tag,
                Float64(τ),
            )
        else
            falses(number_of_gamma)
        end

    println(
        "Previously completed γ values: ",
        gamma_values[completed_gamma],
    )

    println(
        "γ values still requiring calculation: ",
        gamma_values[.!completed_gamma],
    )

    # ========================================================
    # Sweep over every gamma/batch pair in parallel
    # ========================================================

    decoherence_tasks = [
        (gamma_index, fp)
        for gamma_index in eachindex(gamma_values)
        for fp in 1:N_BATCHES
        if !completed_gamma[gamma_index]
    ]

    println()
    println(
        "Running $(length(decoherence_tasks)) gamma/batch tasks " *
        "using $(Threads.nthreads()) Julia threads."
    )

    Threads.@threads for task_index in eachindex(decoherence_tasks)

        gamma_index, fp =
            decoherence_tasks[task_index]

        γ =
            gamma_values[gamma_index]

        println(
            "Running $config_name, target=$target_feature, " *
            "γ=$γ, batch=$fp, " *
            "thread=$(Threads.threadid())"
        )
            # =================================================
            # BLUE:
            #
            # Train on ideal reservoir:
            # γ_train = 0
            #
            # Test with decoherence:
            # γ_test = γ
            # =================================================
            γ_train_vec = sampled_gamma_vector(
                γ,
                nqubit;
                gamma_index=gamma_index,
                batch_index=fp,
                draw_index=1,
            )

            γ_test_vec = sampled_gamma_vector(
                γ,
                nqubit;
                gamma_index=gamma_index,
                batch_index=fp,
                draw_index=2,
            )

            γ_zero_vec = zeros(Float64, nqubit)

            if PRINT_ALL_RATE_VECTORS
                rate_name =
                    DECOHERENCE_MODEL === :dephasing ? "gamma" : "kappa"

                lock(RATE_PRINT_LOCK) do
                    println()
                    println("==============================================")
                    println("ALL QUBIT RATE VALUES")
                    println("rate type      = ", rate_name)
                    println("requested mean = ", γ)
                    println("rate index    = ", gamma_index)
                    println("batch          = ", fp)
                    println("train values   = ", γ_train_vec)
                    println("test values    = ", γ_test_vec)
                    println("train mean     = ", mean(γ_train_vec))
                    println(
                        "train std      = ",
                        std(γ_train_vec; corrected=false),
                    )
                    println("test mean      = ", mean(γ_test_vec))
                    println(
                        "test std       = ",
                        std(γ_test_vec; corrected=false),
                    )
                    println("==============================================")
                end
            end

            if DEBUG_DECOHERENCE_STATE && fp == 1
                DEBUG_DECOHERENCE_RATES[] =
                    Float64.(copy(γ_test_vec))

                DEBUG_DECOHERENCE_PROBABILITIES[] = [
                    decoherence_probability(
                        Float64(τ),
                        rate_q,
                    )
                    for rate_q in γ_test_vec
                ]

                DEBUG_DECOHERENCE_MODEL[] =
                    DECOHERENCE_MODEL

                println()
                println("==============================================")
                println("DECOHERENCE CHANNEL PARAMETERS")
                println("model = ", DECOHERENCE_MODEL)
                println("tau   = ", τ)
                println("rates = ", DEBUG_DECOHERENCE_RATES[])
                println("p     = ", DEBUG_DECOHERENCE_PROBABILITIES[])
                println("==============================================")
            end
            prediction_ideal_angle =
                compute_open_teacher_forced_for_batch(
                    fp,
                    ahead,
                    ctx_circuit,
                    fut_circuit,
                    ctx_target,
                    fut_target,
                    nsystem,
                    nmemory,
                    ntimesteps,
                    nlayers,
                    Us,
                    Float64(τ),
                    ds,
                    B1;
                    γ_train=γ_zero_vec,
                    γ_test=γ_test_vec,
                    lag_rep=LAG_REP,
                    mapping_mode=INPUT_MAPPING_MODE,
                    decoherence_backend=
                        DECOHERENCE_CALCULATION_MODE,
                )

            prediction_ideal =
                circuit_angle_to_zscore_matrix(
                    prediction_ideal_angle,
                    μang_rep,
                    σang_rep;
                    angle_max=ANGLE_MAX,
                )

            target_rmse_trained_ideal[
                gamma_index,
                fp,
            ] = finite_rmse(
                @view(
                    prediction_ideal[
                        target_index,
                        :,
                    ]
                ),
                @view(
                    true_fut_matrix[
                        target_index,
                        :,
                    ]
                ),
            )

            # At γ=0, the two experiments are exactly identical.
            # Avoid performing the same training and prediction twice.
            if iszero(γ)
                target_rmse_trained_with_decoherence[
                    gamma_index,
                    fp,
                ] = target_rmse_trained_ideal[
                    gamma_index,
                    fp,
                ]

                continue
            end

            # =================================================
            # RED:
            #
            # Train with decoherence:
            # γ_train = γ
            #
            # Train and test with independent rate vectors
            # sampled from the same distribution.
            # γ_test = γ
            # =================================================

            prediction_decoherent_angle =
                compute_open_teacher_forced_for_batch(
                    fp,
                    ahead,
                    ctx_circuit,
                    fut_circuit,
                    ctx_target,
                    fut_target,
                    nsystem,
                    nmemory,
                    ntimesteps,
                    nlayers,
                    Us,
                    Float64(τ),
                    ds,
                    B1;
                    γ_train=γ_train_vec,
                    γ_test=γ_test_vec,
                    lag_rep=LAG_REP,
                    mapping_mode=INPUT_MAPPING_MODE,
                    decoherence_backend=
                        DECOHERENCE_CALCULATION_MODE,
                )

            prediction_decoherent =
                circuit_angle_to_zscore_matrix(
                    prediction_decoherent_angle,
                    μang_rep,
                    σang_rep;
                    angle_max=ANGLE_MAX,
                )

            target_rmse_trained_with_decoherence[
                gamma_index,
                fp,
            ] = finite_rmse(
                @view(
                    prediction_decoherent[
                        target_index,
                        :,
                    ]
                ),
                @view(
                    true_fut_matrix[
                        target_index,
                        :,
                    ]
                ),
            )
            
        end  # Threads.@threads loop

        # ========================================================
        # Verify that every gamma completed successfully
        # ========================================================

        for gamma_index in eachindex(gamma_values)

            if completed_gamma[gamma_index]
                continue
            end

            γ =
                gamma_values[gamma_index]

            completed_gamma[gamma_index] =
                all(
                    isfinite,
                    @view(
                        target_rmse_trained_ideal[
                            gamma_index,
                            :,
                        ]
                    ),
                ) &&
                all(
                    isfinite,
                    @view(
                        target_rmse_trained_with_decoherence[
                            gamma_index,
                            :,
                        ]
                    ),
                )

            completed_gamma[gamma_index] ||
                error(
                    "The calculation for γ=$γ finished, but one or " *
                    "more batch RMSE values are nonfinite."
                )
        end

        # Save once after all parallel tasks finish.
        if RESUME_DECOHERENCE
            save_decoherence_resume!(
                resume_path;
                gamma_values=gamma_values,
                target_rmse_trained_ideal=
                    target_rmse_trained_ideal,
                target_rmse_trained_with_decoherence=
                    target_rmse_trained_with_decoherence,
                base_tag=base_tag,
                config_name=config_name,
                target_feature=target_feature,
                input_features=input_features,
                ntimesteps=ntimesteps,
                nmemory=nmemory,
                nsystem=nsystem,
                nqubit=nqubit,
                nlayers=nlayers,
                ds=ds,
                ahead=ahead,
                τ=Float64(τ),
            )

            println("Saved decoherence resume checkpoint:")
            println(resume_path)
        end
    # ========================================================
    # Mean and standard deviation over reservoir batches

    target_ideal_mean =
        fill(NaN, number_of_gamma)

    target_ideal_std =
        fill(NaN, number_of_gamma)

    target_decoherent_mean =
        fill(NaN, number_of_gamma)

    target_decoherent_std =
        fill(NaN, number_of_gamma)

    for gamma_index in 1:number_of_gamma
        target_ideal_mean[gamma_index],
        target_ideal_std[gamma_index] =
            finite_mean_std(
                @view(
                    target_rmse_trained_ideal[
                        gamma_index,
                        :,
                    ]
                )
            )

        target_decoherent_mean[gamma_index],
        target_decoherent_std[gamma_index] =
            finite_mean_std(
                @view(
                    target_rmse_trained_with_decoherence[
                        gamma_index,
                        :,
                    ]
                )
            )
    end

    channel_probability =
        [
            decoherence_probability(τ, γ)
            for γ in gamma_values
        ]

    isdir(FIGURES_DIR) ||
        mkpath(FIGURES_DIR)

    isdir(RESULTS_DIR) ||
        mkpath(RESULTS_DIR)

    # ========================================================
    # Save target results in CSV
    # ========================================================

    output_table = DataFrame(
        dataset=fill(
            String(DATA_MODE),
            number_of_gamma,
        ),

        source=fill(
            source_tag,
            number_of_gamma,
        ),

        configuration=fill(
            config_name,
            number_of_gamma,
        ),

        target=fill(
            target_feature,
            number_of_gamma,
        ),

        input_features=fill(
            feature_tag,
            number_of_gamma,
        ),

        ntimesteps=fill(
            ntimesteps,
            number_of_gamma,
        ),

        nmemory=fill(
            nmemory,
            number_of_gamma,
        ),

        nsystem=fill(
            nsystem,
            number_of_gamma,
        ),

        nqubit=fill(
            nqubit,
            number_of_gamma,
        ),

        nlayers=fill(
            nlayers,
            number_of_gamma,
        ),

        ds=fill(
            ds,
            number_of_gamma,
        ),

        ahead=fill(
            ahead,
            number_of_gamma,
        ),

        tau=fill(
            Float64(τ),
            number_of_gamma,
        ),

        gamma=gamma_values,

        rate_symbol=fill(
            DECOHERENCE_MODEL === :dephasing ? "gamma" : "kappa",
            number_of_gamma,
        ),

        noise_model=fill(
            decoherence_model_name(),
            number_of_gamma,
        ),

        gamma_profile_mode=fill(
            decoherence_profile_mode_name(),
            number_of_gamma,
        ),

        gamma_profile_tag=fill(
            gamma_profile_tag(),
            number_of_gamma,
        ),

        use_gaussian_qubit_rates=fill(
            USE_GAUSSIAN_QUBIT_RATES,
            number_of_gamma,
        ),

        gamma_std_mode=fill(
            String(GAMMA_STD_MODE),
            number_of_gamma,
        ),

        gamma_std_fraction=fill(
            Float64(GAMMA_STD_FRACTION),
            number_of_gamma,
        ),

        gamma_fixed_std=fill(
            Float64(GAMMA_FIXED_STD),
            number_of_gamma,
        ),

        gamma_std_used=[
            gamma_standard_deviation(γ)
            for γ in gamma_values
        ],

        gamma_profile_seed=fill(
            Int(GAMMA_PROFILE_SEED),
            number_of_gamma,
        ),

        channel_probability=
            channel_probability,

        trained_ideal_mean_rmse=
            target_ideal_mean,

        trained_ideal_std_rmse=
            target_ideal_std,

        trained_with_decoherence_mean_rmse=
            target_decoherent_mean,

        trained_with_decoherence_std_rmse=
            target_decoherent_std,
    )

    csv_path = joinpath(
        RESULTS_DIR,
        output_tag * ".csv",
    )

    CSV.write(
        csv_path,
        output_table,
    )

    # ========================================================
    # Save complete per-batch RMSE arrays in JLD2
    # ========================================================

    jld_path = joinpath(
        RESULTS_DIR,
        output_tag * ".jld2",
    )

    train_rate_probability_pack =
        build_rate_probability_tensors(
            gamma_values,
            nqubit,
            Float64(τ);
            draw_index=1,
        )

    test_rate_probability_pack =
        build_rate_probability_tensors(
            gamma_values,
            nqubit,
            Float64(τ);
            draw_index=2,
        )

    jldsave(
        jld_path;

        # Dataset metadata
        DATA_MODE=String(DATA_MODE),
        source_tag=source_tag,
        configuration=config_name,
        target_feature=target_feature,
        target_index=target_index,
        INPUT_FEATURES=input_features,
        feature_names=feature_names,

        # Selected reservoir configuration
        INPUT_MAPPING_MODE=
            String(INPUT_MAPPING_MODE),
        NTIMESTEPS=ntimesteps,
        N_MEMORY=nmemory,
        NSYSTEM=nsystem,
        N_LAYERS=nlayers,
        DS=ds,
        ahead=ahead,
        nqubit=nqubit,

        # Reservoir metadata
        HAM_NAME=HAM_NAME,
        hamiltonian_kind=
            String(HAMILTONIAN_KIND),
        EVOLUTION_TIME_MODE=
            String(EVOLUTION_TIME_MODE),
        τ=Float64(τ),
        lambda=λ,
        number_of_batches=N_BATCHES,
        normalization_mode=
            String(NORM_MODE),

        noise_model=decoherence_model_name(),

        decoherence_calculation_mode=
            String(DECOHERENCE_CALCULATION_MODE),

        decoherence_convention=
            DECOHERENCE_MODEL === :dephasing ?
            "p=1-exp(-4*gamma*tau)" :
            "p=1-exp(-kappa*tau)",

        # Mean-rate list.
        gamma=gamma_values,
        mean_rate=gamma_values,

        rate_symbol=
            DECOHERENCE_MODEL === :dephasing ?
            "gamma" :
            "kappa",

        gamma_is_mean_rate=
            USE_GAUSSIAN_QUBIT_RATES,

        gamma_profile_mode=
            decoherence_profile_mode_name(),

        gamma_profile_tag=
            gamma_profile_tag(),

        use_gaussian_qubit_rates=
            USE_GAUSSIAN_QUBIT_RATES,

        gamma_std_mode=
            String(GAMMA_STD_MODE),

        gamma_std_fraction=
            Float64(GAMMA_STD_FRACTION),

        gamma_fixed_std=
            Float64(GAMMA_FIXED_STD),

        gamma_std_used=[
            gamma_standard_deviation(γ)
            for γ in gamma_values
        ],

        gamma_profile_seed=
            Int(GAMMA_PROFILE_SEED),

        rate_vector_shape=
            "gamma_index × batch_index × qubit_index",

        # Probability at the listed mean rate.
        channel_probability=
            channel_probability,

        channel_probability_at_mean_rate=
            channel_probability,

        # Keep this only for backward compatibility with older scripts.
        phase_damping_probability=
            channel_probability,

        # Backward-compatible scalar fields.
        blue_training_rate=0.0,
        blue_test_rate=
            gamma_values,

        red_training_rate=
            gamma_values,

        red_test_rate=
            gamma_values,

        # Actual qubit-dependent rates used in the calculation.
        blue_training_rate_vectors=
            train_rate_probability_pack.zero_rate_tensor,

        blue_test_rate_vectors=
            test_rate_probability_pack.rate_tensor,

        red_training_rate_vectors=
            train_rate_probability_pack.rate_tensor,

        red_test_rate_vectors=
            test_rate_probability_pack.rate_tensor,

        # Actual qubit-dependent channel probabilities.
        blue_training_channel_probability_vectors=
            train_rate_probability_pack.zero_probability_tensor,

        blue_test_channel_probability_vectors=
            test_rate_probability_pack.probability_tensor,

        red_training_channel_probability_vectors=
            train_rate_probability_pack.probability_tensor,

        red_test_channel_probability_vectors=
            test_rate_probability_pack.probability_tensor,

        qubit_rate_mean_per_gamma_batch=
            test_rate_probability_pack.rate_mean_per_gamma_batch,

        qubit_rate_std_per_gamma_batch=
            test_rate_probability_pack.rate_std_per_gamma_batch,

        qubit_rate_min_per_gamma_batch=
            test_rate_probability_pack.rate_min_per_gamma_batch,

        qubit_rate_max_per_gamma_batch=
            test_rate_probability_pack.rate_max_per_gamma_batch,

        qubit_probability_mean_per_gamma_batch=
            test_rate_probability_pack.probability_mean_per_gamma_batch,

        qubit_probability_std_per_gamma_batch=
            test_rate_probability_pack.probability_std_per_gamma_batch,

        qubit_probability_min_per_gamma_batch=
            test_rate_probability_pack.probability_min_per_gamma_batch,

        qubit_probability_max_per_gamma_batch=
            test_rate_probability_pack.probability_max_per_gamma_batch,

        # Complete target RMSE for every batch
        target_rmse_trained_ideal=
            target_rmse_trained_ideal,

        target_rmse_trained_with_decoherence=
            target_rmse_trained_with_decoherence,

        # Mean and standard deviation
        target_ideal_mean=
            target_ideal_mean,

        target_ideal_std=
            target_ideal_std,

        target_decoherent_mean=
            target_decoherent_mean,

        target_decoherent_std=
            target_decoherent_std,
    )

    # ========================================================
    # Plot selected target
    # ========================================================

    ideal_plot_values = [
        isfinite(value) && value > 0 ?
            value :
            NaN
        for value in target_ideal_mean
    ]

    decoherent_plot_values = [
        isfinite(value) && value > 0 ?
            value :
            NaN
        for value in target_decoherent_mean
    ]

    ideal_upper_error = [
        isfinite(value) ?
            value :
            0.0
        for value in target_ideal_std
    ]

    decoherent_upper_error = [
        isfinite(value) ?
            value :
            0.0
        for value in target_decoherent_std
    ]

    ideal_lower_error = [
        isfinite(target_ideal_mean[index]) &&
        target_ideal_mean[index] > 0 ?
            min(
                ideal_upper_error[index],
                0.95 * target_ideal_mean[index],
            ) :
            0.0
        for index in eachindex(target_ideal_mean)
    ]

    decoherent_lower_error = [
        isfinite(target_decoherent_mean[index]) &&
        target_decoherent_mean[index] > 0 ?
            min(
                decoherent_upper_error[index],
                0.95 * target_decoherent_mean[index],
            ) :
            0.0
        for index in eachindex(target_decoherent_mean)
    ]

    figure = plot(
        gamma_values,
        ideal_plot_values;
        yerror=(
            ideal_lower_error,
            ideal_upper_error,
        ),
        yscale=:log10,
        xlabel=DECOHERENCE_MODEL === :dephasing ? "γ" : "κ",
        ylabel="RMSE",
        label="Trained on ideal reservoir",
        linewidth=2,
        marker=:diamond,
        markersize=5,
        markercolor=:white,
        markerstrokecolor=:blue,
        seriescolor=:blue,
        legend=:bottomright,
        size=(720, 480),
        title=@sprintf(
            "%s | %s | %s → %s | I%dM%d | ds=%d",
            String(DATA_MODE),
            decoherence_model_name(),
            config_name,
            target_feature,
            ntimesteps,
            nmemory,
            ds,
        ),
    )

    plot!(
        figure,
        gamma_values,
        decoherent_plot_values;
        yerror=(
            decoherent_lower_error,
            decoherent_upper_error,
        ),
        label="Trained with decoherence",
        linewidth=2,
        marker=:circle,
        markersize=5,
        markercolor=:white,
        markerstrokecolor=:red,
        seriescolor=:red,
    )

    figure_path = joinpath(
        FIGURES_DIR,
        output_tag * ".png",
    )

    # savefig(
    #     figure,
    #     figure_path,
    # )

    println()
    println("Saved decoherence results:")
    println("CSV:    $csv_path")
    println("JLD2:   $jld_path")
    println("Figure: $figure_path")

    return output_table
end

function run_selected_decoherence_jobs()
    DATA_MODE in (:theory, :experiment) ||
        error(
            "The selected u1/u2/iB/iC jobs support only " *
            "DATA_MODE=:theory or DATA_MODE=:experiment."
        )

    active_jobs = [
        job for job in DECOHERENCE_JOBS
        if job.data_mode === DATA_MODE &&
        job.config == "4to1"
    ]

    isempty(active_jobs) &&
        error(
            "No decoherence jobs were defined for " *
            "DATA_MODE=$DATA_MODE."
        )

    isdir(RESULTS_DIR) ||
        mkpath(RESULTS_DIR)

    all_tables =
        DataFrame[]

    number_of_jobs =
        length(active_jobs)

    for (job_index, job) in
        enumerate(active_jobs)

        println()
        println("##############################################")
        println(
            "DECOHERENCE JOB " *
            "$job_index / $number_of_jobs"
        )
        println("configuration = $(job.config)")
        println("target        = $(job.target)")
        println("features      = $(job.features)")
        println("I             = $(job.ntimesteps)")
        println("M             = $(job.nmemory)")
        println("ds            = $(job.ds)")
        println("##############################################")

        result_table =
            run_decoherence_sweep(
                config_name=job.config,
                target_feature=job.target,
                input_features=job.features,
                ntimesteps=job.ntimesteps,
                nmemory=job.nmemory,
                ds=job.ds,
                nlayers=1,
                ahead=0,
            )

        push!(
            all_tables,
            result_table,
        )
    end

    combined_table =
        vcat(
            all_tables...;
            cols=:union,
        )

    combined_source_tag =
        if DATA_MODE === :theory
            "theory"
        else
            @sprintf(
                "experiment_Vs%.2f_board%d",
                TARGET_VS,
                BOARD_ID,
            )
        end

    combined_path = joinpath(
        RESULTS_DIR,
        @sprintf(
            "decoherence_selected_configs_%s_noise%s_calc%s_%s_ham_%s_tauMode_%s_batch%d_g%.3f-%.3f_ngamma%d.csv",
            combined_source_tag,
            decoherence_model_name(),
            String(DECOHERENCE_CALCULATION_MODE),
            gamma_profile_tag(),
            HAM_NAME,
            String(EVOLUTION_TIME_MODE),
            N_BATCHES,
            first(GAMMA_LIST),
            last(GAMMA_LIST),
            length(GAMMA_LIST),
        ),
    )

    CSV.write(
        combined_path,
        combined_table,
    )

    println()
    println(
        "Saved combined selected-configuration CSV:"
    )
    println(combined_path)

    return combined_table
end

function run_one_selected_decoherence_job(
    config_name::String,
    target_feature::String,
)
    job_index = findfirst(
        job ->
            job.data_mode === DATA_MODE &&
            job.config == config_name &&
            job.target == target_feature,
        DECOHERENCE_JOBS,
    )

    job_index === nothing &&
        error(
            "No decoherence job exists for " *
            "DATA_MODE=$DATA_MODE, configuration=$config_name, " *
            "target=$target_feature."
        )

    job = DECOHERENCE_JOBS[job_index]

    println()
    println("Selected decoherence job:")
    println("DATA_MODE     = $(job.data_mode)")
    println("configuration = $(job.config)")
    println("target        = $(job.target)")
    println("features      = $(job.features)")
    println("I             = $(job.ntimesteps)")
    println("M             = $(job.nmemory)")
    println("ds            = $(job.ds)")

    return run_decoherence_sweep(
        config_name=job.config,
        target_feature=job.target,
        input_features=job.features,
        ntimesteps=job.ntimesteps,
        nmemory=job.nmemory,
        ds=job.ds,
        nlayers=1,
        ahead=0,
    )
end
# -------------------- MAIN: run sweeps (OPEN only) --------------------
function run_multi_mstep_prediction()
    data_file = data_file_for(
        BOARD_ID,
        TARGET_RUN_ID,
    )

    isfile(data_file) ||
        error("Data file not found: $data_file")

    println(
        "Loading raw data, normalizing once, " *
        "then preparing representation..."
    )

    data_pack = load_and_prepare_data(
        data_file;
        norm=NORM_MODE,
        input_features=INPUT_FEATURES,
        maplike=MAPLIKE,
        maplike_which=MAPLIKE_WHICH,
        maplike_target=MAPLIKE_TARGET,
        apply_lowpass=DATA_MODE === :experiment,
        default_cutoff_Hz=DEFAULT_CUTOFF_HZ,
        default_filter_order=DEFAULT_FILTER_ORDER,
        u2_cutoff_Hz=U2_CUTOFF_HZ,
        u2_filter_order=U2_FILTER_ORDER,
    )

    ctx_angle_orig = data_pack.ctx_used
    fut_angle_orig = data_pack.fut_used
    normμ = data_pack.μ
    normσ = data_pack.σ
    t_axis = data_pack.t_axis

    # These are always normalized RAW waveform arrays.
    ctx_zscore_raw = data_pack.ctx_zscore_raw
    fut_zscore_raw = data_pack.fut_zscore_raw
    ctx_zscore_eval_orig = data_pack.ctx_zscore_eval
    fut_zscore_eval_orig = data_pack.fut_zscore_eval

    t_ctx_raw = data_pack.t_ctx_raw
    t_fut_raw = data_pack.t_fut_raw
    ctx_target_orig = data_pack.ctx_target
    fut_target_orig = data_pack.fut_target

    ctx_angle = repeat_features_matrix(ctx_angle_orig, REPEAT_FEATURES)
    fut_angle = repeat_features_matrix(fut_angle_orig, REPEAT_FEATURES)

    ctx_circuit = ctx_angle
    fut_circuit = fut_angle

    ctx_target = repeat_features_matrix(ctx_target_orig, REPEAT_FEATURES)
    fut_target = repeat_features_matrix(fut_target_orig, REPEAT_FEATURES)

    μang_rep = repeat(data_pack.μang, REPEAT_FEATURES)
    σang_rep = repeat(data_pack.σang, REPEAT_FEATURES)


    if size(fut_angle, 2) == 0
        error(
            "No future data in interval $(ACTIVE_T_FUT) $TIME_UNIT. " *
            "Check the CSV time range and selected future interval."
        )
    end

    F = size(ctx_angle, 1)
    N_fut = size(fut_angle, 2)

    # z-score target for plotting/RMSE
    fut_plot = repeat_features_matrix(fut_zscore_eval_orig, REPEAT_FEATURES)

    isdir(FIGURES_DIR) || mkpath(FIGURES_DIR)
    isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
    isdir(PRECOMP_U_DIR) || mkpath(PRECOMP_U_DIR)


    feature_str = join(INPUT_FEATURES, "-")
    max_nn = MAX_NEAREST_NEIGHBOR

    sanitize_fname(s::AbstractString) = replace(s, r"[^\w\-\[\],.=]+" => "_")

    line2 = @sprintf(
        "CTX=[%.1f–%.1f] %s, FUT=[%.1f–%.1f] %s, Batches=%d",
        ACTIVE_T_CTX[1],
        ACTIVE_T_CTX[2],
        TIME_UNIT,
        ACTIVE_T_FUT[1],
        ACTIVE_T_FUT[2],
        TIME_UNIT,
        N_BATCHES,
    )


    

    for target_nqubit in MAX_TOTAL_QUBITS:-1:1
        @info "Starting sweep for total nqubit = $target_nqubit"

        for nmemory in MEMORY_LIST
            for ntimesteps in TIMESTEPS_PER_QUBIT_LIST
                for nlayers in LAYERS_LIST
                    for ds in DS_LIST
                        nsystem_run = effective_nsystem(INPUT_MAPPING_MODE, ntimesteps, LAG_REP)
                        nqubit = nsystem_run + nmemory

                        if nqubit != target_nqubit
                            continue
                        end

                        min_nqubit = 2
                        if nqubit < min_nqubit
                            @warn "Skipping case: mem=$nmemory, nt=$ntimesteps, layers=$nlayers, ds=$ds, nsystem=$nsystem_run, nqubit=$nqubit (minimum required qubits = $min_nqubit)"
                            continue
                        end


                        τ = evolution_time(nqubit)

                        line1 = @sprintf(
                            "Memory=%d, Mode=%s, NTimesteps=%d, Layers=%d, ds=%d, MaxNN=%d, λ=%s, τ=%.1f, nsystem=%d",
                            nmemory, String(INPUT_MAPPING_MODE), ntimesteps, nlayers, ds, max_nn,
                            lambda_str(λ), τ, nsystem_run
                        )
                        meta_str_common = line1 * "\n" * line2

                        B1 = nothing
                        Hs = nothing
                        Us = nothing

                        function tag_common(ahead::Int, τ::Real, feature_str::String, nsystem_run::Int, ntimesteps::Int, nlayers::Int)
                            mode_str = MAPLIKE ? "maplike" : "raw"
                            which_str = MAPLIKE ? "which_" * String(MAPLIKE_WHICH) : ""
                            target_str = MAPLIKE ? "_target_" * String(MAPLIKE_TARGET) : ""
                            
                            source_str = if DATA_MODE === :theory
                                "theory"

                            elseif DATA_MODE === :experiment
                                @sprintf(
                                    "experiment_Vs%.2f_board%d",
                                    TARGET_VS,
                                    BOARD_ID,
                                )

                            else
                                error(
                                    "DATA_MODE must be :theory or :experiment."
                                )
                            end
                            

                            s = @sprintf(
                                "%s%s%s_ham%s_m%02d_%s_features_%s_ctx[%.0f,%.0f]_fut[%.0f,%.0f]_mem%d_mode%s_nt%d_layers%d_λ%s_τ%.1f_lag%d_ds%d_rep%d_nsys%d_%s_batch%d",
                                mode_str,
                                MAPLIKE ? "_" * which_str : "",
                                target_str,
                                HAM_NAME,
                                ahead,
                                source_str,
                                feature_str,
                                ACTIVE_T_CTX[1],
                                ACTIVE_T_CTX[2],
                                ACTIVE_T_FUT[1],
                                ACTIVE_T_FUT[2],
                                nmemory, String(INPUT_MAPPING_MODE), ntimesteps, nlayers,
                                lambda_str(λ), τ, LAG_REP, ds, REPEAT_FEATURES,
                                nsystem_run,
                                String(APPROX_METHOD), N_BATCHES
                            )
                            return sanitize_fname(s)
                        end

                        for ahead in HORIZON_LIST
                            tag = tag_common(ahead, τ, feature_str, nsystem_run, ntimesteps, nlayers)
                            out_prefix = RUN_CLOSED_LOOP ? closed_loop_file_prefix() : "open_teacherforced_only_"
                            outpath = joinpath(RESULTS_DIR, out_prefix * tag * ".jld2")

                            jld_exists = isfile(outpath)

                            if SKIP_IF_RESULT_EXISTS && jld_exists
                                missing = missing_figure_keys(tag)

                                if isempty(missing)
                                    @info "Skipping: JLD and all figures already exist for mem=$nmemory, nt=$ntimesteps, layers=$nlayers, ds=$ds, horizon=$ahead"
                                else
                                    @info "JLD exists, but some figures are missing. Regenerating missing figures only."
                                    @info "Missing figure keys: $missing"

                                    regenerate_missing_figures_from_jld!(
                                        outpath,
                                        tag,
                                        meta_str_common,
                                    )
                                end

                                continue
                            end
                                # Load/build only when needed
                            if B1 === nothing
                                B1 = build_measurement_ops(
                                    INPUT_FEATURES,
                                    nsystem_run,
                                    nmemory,
                                )

                                @info(
                                    "Pauli measurement reservoir",
                                    number_of_observables=length(B1),
                                    max_nearest_neighbor=MAX_NEAREST_NEIGHBOR,
                                )
                            end

                            if Us === nothing
                                Hs, Us = get_or_generate_Hs_Us(
                                    nqubit,
                                    τ;
                                    Batch=N_BATCHES,
                                    seed=42,
                                )
                            end    # Load/build only when needed


                            println("\n==============================")
                            println(RUN_CLOSED_LOOP ?
                                "Running CLOSED: mem=$nmemory, nt=$ntimesteps, layers=$nlayers, ds=$ds, horizon=$ahead" :
                                "Running OPEN: mem=$nmemory, nt=$ntimesteps, layers=$nlayers, ds=$ds, horizon=$ahead")
                            println("==============================")

                            open_pred_angle_all =
                                RUN_CLOSED_LOOP ?
                                nothing :
                                Vector{Array{Float64,2}}(undef, N_BATCHES)

                            closed_pred_angle_all =
                                RUN_CLOSED_LOOP ?
                                Vector{Array{Float64,2}}(undef, N_BATCHES) :
                                nothing

                            open_pred_zscore_all =
                                RUN_CLOSED_LOOP ?
                                nothing :
                                Vector{Array{Float64,2}}(undef, N_BATCHES)

                            closed_pred_zscore_all =
                                RUN_CLOSED_LOOP ?
                                Vector{Array{Float64,2}}(undef, N_BATCHES) :
                                nothing

                            train_inputs_override = nothing
                            train_targets_override = nothing

                                
                            Threads.@threads for fp in 1:N_BATCHES
                                if RUN_CLOSED_LOOP
                                    if RUN_BLOCKWISE_CLOSED_LOOP && ahead <= 0
                                        error("RUN_BLOCKWISE_CLOSED_LOOP=true requires horizon > 0 because horizon is used as the block size.")
                                    end

                                    closed_pred_angle_all[fp] = if RUN_BLOCKWISE_CLOSED_LOOP
                                        compute_blockwise_closed_loop_for_batch(
                                            fp, ahead,
                                            ctx_circuit, fut_circuit, ctx_target, fut_target,
                                            μang_rep, σang_rep,
                                            nsystem_run, nmemory, ntimesteps, nlayers,
                                            Us, τ, ds,
                                            B1;
                                            lag_rep=LAG_REP,
                                            mapping_mode=INPUT_MAPPING_MODE,
                                        )
                                    else
                                        compute_closed_loop_for_batch(
                                            fp, ahead,
                                            ctx_circuit, fut_circuit, ctx_target, fut_target,
                                            μang_rep, σang_rep,
                                            nsystem_run, nmemory, ntimesteps, nlayers,
                                            Us, τ, ds,
                                            B1;
                                            lag_rep=LAG_REP,
                                            mapping_mode=INPUT_MAPPING_MODE,
                                        )
                                    end
                                else
                                    open_pred_angle_all[fp] =
                                        compute_open_teacher_forced_for_batch(
                                            fp,
                                            ahead,
                                            ctx_circuit,
                                            fut_circuit,
                                            ctx_target,
                                            fut_target,
                                            nsystem_run,
                                            nmemory,
                                            ntimesteps,
                                            nlayers,
                                            Us,
                                            τ,
                                            ds,
                                            B1;
                                            lag_rep=LAG_REP,
                                            mapping_mode=INPUT_MAPPING_MODE,
                                            train_inputs_override=train_inputs_override,
                                            train_targets_override=train_targets_override,
                                        )
                                end
                            end
                            last_ctx_extrema_time = data_pack.last_ctx_extrema_time
                            if !RUN_CLOSED_LOOP
                                for fp in 1:N_BATCHES
                                    open_pred_zscore_all[fp] =
                                        circuit_angle_to_zscore_matrix(
                                            open_pred_angle_all[fp],
                                            μang_rep,
                                            σang_rep;
                                            angle_max=ANGLE_MAX,
                                        )
                                end
                            else
                                for fp in 1:N_BATCHES
                                    closed_pred_zscore_all[fp] = circuit_angle_to_zscore_matrix(
                                        closed_pred_angle_all[fp],
                                        μang_rep,
                                        σang_rep;
                                        angle_max = ANGLE_MAX,
                                    )
                                end
                            end

                            open_pred_timepos_all = nothing
                            open_pred_gapraw_all = nothing

                            if !RUN_CLOSED_LOOP && MAPLIKE && MAPLIKE_TARGET === :time
                                # Convert predicted z-score Δt back to physical Δt.
                                # Store both raw Δt and reconstructed extrema times.
                                open_pred_timepos_all = Vector{Array{Float64,2}}(undef, N_BATCHES)
                                open_pred_gapraw_all  = Vector{Array{Float64,2}}(undef, N_BATCHES)

                                μgap_rep = repeat(data_pack.μgap, REPEAT_FEATURES)
                                σgap_rep = repeat(data_pack.σgap, REPEAT_FEATURES)

                                for fp in 1:N_BATCHES
                                    Ygap_z = open_pred_zscore_all[fp]
                                    Ygap_raw = Array{Float64}(undef, size(Ygap_z))

                                    @inbounds for s in axes(Ygap_z, 2)
                                        Ygap_raw[:, s] .= Ygap_z[:, s] .* σgap_rep .+ μgap_rep
                                    end

                                    open_pred_gapraw_all[fp] = Ygap_raw

                                    Ypos = fill(NaN, size(Ygap_raw))
                                    
                                    for f in 1:feature_size
                                        t0 = last_ctx_extrema_time[f]

                                        if isfinite(t0)
                                            Ypos_raw = gaps_to_times(vec(Ygap_raw[f, :]), t0)

                                            # Align first predicted FUT extremum to first true FUT extremum.
                                            # In MAPLIKE_TARGET=:time, t_axis is the selected FUT extrema time axis.
                                            if !isempty(t_axis) && !isempty(Ypos_raw) && isfinite(Ypos_raw[1])
                                                shift = t_axis[1] - Ypos_raw[1]
                                                Ypos[f, :] .= Ypos_raw .+ shift
                                            else
                                                Ypos[f, :] .= Ypos_raw
                                            end
                                        end
                                    end

                                    open_pred_timepos_all[fp] = Ypos
                                end
                            end
                            # --- RMSE vs time over FUT: mean over batches, per feature (NORMALIZED) ---
                            open_rmse_vs_time = RUN_CLOSED_LOOP ? nothing : fill(NaN, feature_size, N_fut)

                            if !RUN_CLOSED_LOOP
                                for f in 1:feature_size
                                    for s in 1:N_fut
                                        errs_open_sq = Float64[]
                                        y_true = fut_plot[f, s]

                                        for fp in 1:N_BATCHES
                                            yhatO = open_pred_zscore_all[fp][f, s]
                                            if isfinite(yhatO)
                                                push!(errs_open_sq, (yhatO - y_true)^2)
                                            end
                                        end

                                        open_rmse_vs_time[f, s] = isempty(errs_open_sq) ? NaN : sqrt(mean(errs_open_sq))
                                    end
                                end
                            end
                            closed_rmse_vs_time = RUN_CLOSED_LOOP ? fill(NaN, feature_size, N_fut) : nothing

                            if RUN_CLOSED_LOOP
                                for f in 1:feature_size
                                    for s in 1:N_fut
                                        errs_closed_sq = Float64[]
                                        y_true = fut_plot[f, s]

                                        for fp in 1:N_BATCHES
                                            yhatC = closed_pred_zscore_all[fp][f, s]
                                            if isfinite(yhatC)
                                                push!(errs_closed_sq, (yhatC - y_true)^2)
                                            end
                                        end

                                        closed_rmse_vs_time[f, s] =
                                            isempty(errs_closed_sq) ? NaN : sqrt(mean(errs_closed_sq))
                                    end
                                end
                            end
                            # --- avg RMSE over all batches (per feature), over whole FUT window ---
                            if !RUN_CLOSED_LOOP
                                # --- avg RMSE over all batches (per feature), over whole FUT window ---
                                open_rmse_per_batch = fill(NaN, feature_size, N_BATCHES)
                                avg_open_rmse = fill(NaN, feature_size)

                                for f in 1:feature_size
                                    for fp in 1:N_BATCHES
                                        yhat_norm = open_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- fut_plot[f, :]
                                        m = isfinite.(d)
                                        open_rmse_per_batch[f, fp] = any(m) ? sqrt(mean(abs2, d[m])) : NaN
                                    end

                                    vals = view(open_rmse_per_batch, f, :)
                                    good = isfinite.(vals)
                                    avg_open_rmse[f] = any(good) ? mean(vals[good]) : NaN
                                end

                                open_rmse_overall_per_batch = fill(NaN, N_BATCHES)

                                for fp in 1:N_BATCHES
                                    errs_sq = Float64[]

                                    for f in 1:feature_size
                                        yhat_norm = @view open_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- @view(fut_plot[f, :])
                                        m = isfinite.(d)

                                        if any(m)
                                            append!(errs_sq, abs2.(d[m]))
                                        end
                                    end

                                    open_rmse_overall_per_batch[fp] = isempty(errs_sq) ? NaN : sqrt(mean(errs_sq))
                                end

                                good_overall = isfinite.(open_rmse_overall_per_batch)

                                rmse_all_features_all_batches =
                                    any(good_overall) ? mean(open_rmse_overall_per_batch[good_overall]) : NaN

                                best_fp_open_together =
                                    any(good_overall) ? argmin(open_rmse_overall_per_batch) : 0

                                best_open_score_together =
                                    best_fp_open_together == 0 ? NaN : open_rmse_overall_per_batch[best_fp_open_together]

                                best_fp_open_per_feature = fill(0, feature_size)
                                best_open_score = fill(NaN, feature_size)

                                for f in 1:feature_size
                                    best_val = Inf
                                    best_fp = 0

                                    for fp in 1:N_BATCHES
                                        yhat_norm = open_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- fut_plot[f, :]
                                        m = isfinite.(d)
                                        val = any(m) ? sqrt(mean(abs2, d[m])) : Inf

                                        if val < best_val
                                            best_val = val
                                            best_fp = fp
                                        end
                                    end

                                    best_fp_open_per_feature[f] = best_fp
                                    best_open_score[f] = isfinite(best_val) ? best_val : NaN
                                end
                            else
                                open_rmse_per_batch = nothing
                                avg_open_rmse = fill(NaN, feature_size)
                                open_rmse_overall_per_batch = nothing
                                rmse_all_features_all_batches = NaN
                                best_fp_open_together = 0
                                best_open_score_together = NaN
                                best_fp_open_per_feature = fill(0, feature_size)
                                best_open_score = fill(NaN, feature_size)
                            end
                            best_fp_closed_per_feature = RUN_CLOSED_LOOP ? fill(0, feature_size) : nothing
                            best_closed_score = RUN_CLOSED_LOOP ? fill(NaN, feature_size) : nothing

                            # Mean per-feature RMSE over batches.
                            # This is the quantity comparable to the heatmap for each feature.
                            avg_closed_rmse = fill(NaN, feature_size)
                            closed_rmse_per_batch = RUN_CLOSED_LOOP ? fill(NaN, feature_size, N_BATCHES) : nothing

                            if RUN_CLOSED_LOOP
                                for f in 1:feature_size
                                    vals = Float64[]

                                    for fp in 1:N_BATCHES
                                        yhat_norm = closed_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- fut_plot[f, :]
                                        m = isfinite.(d)

                                        r = any(m) ? sqrt(mean(abs2, d[m])) : NaN
                                        closed_rmse_per_batch[f, fp] = r

                                        if isfinite(r)
                                            push!(vals, r)
                                        end
                                    end

                                    avg_closed_rmse[f] = isempty(vals) ? NaN : mean(vals)
                                end
                            end

                            closed_rmse_overall_per_batch = RUN_CLOSED_LOOP ? fill(NaN, N_BATCHES) : nothing
                            rmse_closed_all_features_all_batches = NaN
                            best_fp_closed_together = 0
                            best_closed_score_together = NaN

                            if RUN_CLOSED_LOOP
                                for fp in 1:N_BATCHES
                                    errs_sq = Float64[]

                                    for f in 1:feature_size
                                        yhat_norm = @view closed_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- @view(fut_plot[f, :])
                                        m = isfinite.(d)

                                        if any(m)
                                            append!(errs_sq, abs2.(d[m]))
                                        end
                                    end

                                    closed_rmse_overall_per_batch[fp] =
                                        isempty(errs_sq) ? NaN : sqrt(mean(errs_sq))
                                end

                                good_closed_overall = isfinite.(closed_rmse_overall_per_batch)

                                rmse_closed_all_features_all_batches =
                                    any(good_closed_overall) ? mean(closed_rmse_overall_per_batch[good_closed_overall]) : NaN

                                best_fp_closed_together =
                                    any(good_closed_overall) ? argmin(closed_rmse_overall_per_batch) : 0

                                best_closed_score_together =
                                    best_fp_closed_together == 0 ? NaN : closed_rmse_overall_per_batch[best_fp_closed_together]
                            end
                            if RUN_CLOSED_LOOP
                                for f in 1:feature_size
                                    best_val = Inf
                                    best_fp = 0

                                    for fp in 1:N_BATCHES
                                        yhat_norm = closed_pred_zscore_all[fp][f, :]
                                        d = yhat_norm .- fut_plot[f, :]
                                        m = isfinite.(d)
                                        val = any(m) ? sqrt(mean(abs2, d[m])) : Inf

                                        if val < best_val
                                            best_val = val
                                            best_fp = fp
                                        end
                                    end

                                    best_fp_closed_per_feature[f] = best_fp
                                    best_closed_score[f] = isfinite(best_val) ? best_val : NaN
                                end
                            end

                            # These paths are still saved in the JLD2 as `nothing`
                            # when figure generation is disabled.
                            rmse_open_path = nothing
                            rmse_closed_path = nothing

                            ts_open_path = nothing
                            ts_closed_path = nothing

                            ts_open_together_path = nothing
                            ts_closed_together_path = nothing

                            ts_open_ctx_path = nothing
                            ts_closed_ctx_path = nothing

                            if MAKE_PREDICTION_FIGURES

                                c_open = :orange
                                c_closed = :green
                                c_true = :blue
                                legend_fs = 9
                            
                                legend_bg = RGBA(1, 1, 1, 0.55)
                                # ---------------- COMMON PLOT WINDOW ----------------
                                # Use the same time range for RMSE-vs-time and time-series figures.
                                t_min, t_max = PLOT_T_RANGE

                                mask = (t_axis .>= t_min) .& (t_axis .<= t_max)
                                mask_raw = (t_fut_raw .>= t_min) .& (t_fut_raw .<= t_max)
                                # ---------------- FIGURE A: OPEN RMSE vs time ----------------
                                rmse_open_path = nothing

                                if !RUN_CLOSED_LOOP
                                    p_rmse_open = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    for f in 1:feature_size
                                        plot!(p_rmse_open[f], t_axis[mask], @view(open_rmse_vs_time[f, mask]);
                                            lw=2,
                                            seriescolor=c_open,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label="Open (mean over batches)",
                                            ylabel=INPUT_FEATURES[f] * " RMSE (norm)")
                                        title!(p_rmse_open[f], @sprintf("%s  (OPEN)", INPUT_FEATURES[f]))
                                        plot!(p_rmse_open[f]; legend=:topright)
                                    end
                                    xlabel!(p_rmse_open[feature_size], time_axis_label())

                                    idx_meta = feature_size + 1
                                    plot!(p_rmse_open[idx_meta], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)
                                    annotate!(p_rmse_open[idx_meta], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("OPEN, Horizon=%d", ahead), 14, :center))

                                    fname_rmse_open = "rmse_vsTime_OPEN_MEANBATCH_" * tag * ".png"
                                    rmse_open_path = joinpath(FIGURES_DIR, fname_rmse_open)
                                    # savefig(p_rmse_open, rmse_open_path)
                                    @info "Saved OPEN RMSE-vs-time figure to $rmse_open_path"
                                end
                                rmse_closed_path = nothing

                                if RUN_CLOSED_LOOP
                                    p_rmse_closed = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    for f in 1:feature_size
                                        plot!(p_rmse_closed[f], t_axis[mask], @view(closed_rmse_vs_time[f, mask]);
                                            lw=2,
                                            seriescolor=c_closed,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label=closed_loop_label() * " (mean over batches)",
                                            ylabel=INPUT_FEATURES[f] * " RMSE (norm)")
                                        title!(p_rmse_closed[f], @sprintf("%s  (%s)", INPUT_FEATURES[f], closed_loop_mode_name()))
                                        plot!(p_rmse_closed[f]; legend=:topright)
                                    end

                                    xlabel!(p_rmse_closed[feature_size], time_axis_label())

                                    idx_meta_closed = feature_size + 1
                                    plot!(p_rmse_closed[idx_meta_closed], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)

                                    annotate!(p_rmse_closed[idx_meta_closed], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("%s, Horizon=%d", closed_loop_mode_name(), ahead), 14, :center))

                                    fname_rmse_closed = "rmse_vsTime_" * closed_loop_mode_name() * "_MEANBATCH_" * tag * ".png"
                                    rmse_closed_path = joinpath(FIGURES_DIR, fname_rmse_closed)

                                    # savefig(p_rmse_closed, rmse_closed_path)
                                    @info "Saved CLOSED RMSE-vs-time figure to $rmse_closed_path"
                                end

                                # ---------------- FIGURE B: OPEN Time-series (best batch per feature) ----------------
                                c_true = :blue
                                c_closed = :green
                                legend_fs = 9
                                legend_bg = RGBA(1, 1, 1, 0.55)
                                ts_open_path = nothing

                                if !RUN_CLOSED_LOOP
                                    p_ts_open = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    # --- Restrict time window for Figure B ---

                                    for f in 1:feature_size
                                        fpO = best_fp_open_per_feature[f]
                                        if MAPLIKE && MAPLIKE_TARGET === :time
                                            # fut_plot is z-score Δt. Convert it back to physical Δt.
                                            y_true_gap = fut_plot[f, :] .* σgap_rep[f] .+ μgap_rep[f]
                                            y_open_gap = fill(NaN, N_fut)

                                            if fpO != 0
                                                y_open_gap .= @view open_pred_gapraw_all[fpO][f, :]
                                            end

                                            plot!(p_ts_open[f], t_axis[mask], y_true_gap[mask];
                                                lw=2,
                                                marker=:circle,
                                                markerstrokecolor=:auto,
                                                markersize=2,
                                                seriescolor=:blue,
                                                label="True Δt")

                                            plot!(p_ts_open[f], t_axis[mask], y_open_gap[mask];
                                                lw=2,
                                                marker=:circle,
                                                markerstrokecolor=:auto,
                                                markersize=2,
                                                seriescolor=:orange,
                                                label = fpO == 0 ? "Predicted Δt" :
                                                    @sprintf("Predicted Δt b%02d, RMSE %.1e", fpO, avg_open_rmse[f]))

                                            ylabel!(p_ts_open[f], INPUT_FEATURES[f] * " Δt ($TIME_UNIT)")
                                        else
                                            y_true_f = @view fut_plot[f, :]
                                            y_open_f = fill(NaN, N_fut)

                                            if fpO != 0
                                                y_open_f .= @view open_pred_zscore_all[fpO][f, :]
                                            end

                                            if MAPLIKE
                                                plot!(p_ts_open[f], t_fut_raw[mask_raw], vec(fut_zscore_raw[f, :])[mask_raw];
                                                    lw=2, seriescolor=:black, seriesalpha=0.20,
                                                    label="Actual")

                                                plot!(p_ts_open[f], t_axis[mask], y_true_f[mask];
                                                    lw=0, marker=:utriangle, markersize=6,
                                                    seriescolor=:blue,
                                                    label="Extrema")
                                            end

                                            plot!(p_ts_open[f], t_axis[mask], y_true_f[mask];
                                                lw=2, seriescolor=c_true, marker=:circle, markerstrokecolor=:auto, markersize=2,
                                                label="True")

                                            plot!(
                                                p_ts_open[f],
                                                t_axis[mask],
                                                y_open_f[mask];
                                                lw=2,
                                                seriescolor=c_open,
                                                marker=:circle,
                                                markerstrokecolor=:auto,
                                                markersize=2,
                                                label=fpO == 0 ? "Open" :
                                                    @sprintf(
                                                        "Open b%02d, RMSE mean over batches %.1e",
                                                        fpO,
                                                        avg_open_rmse[f],
                                                    ),
                                            )

                                            ylabel!(
                                                p_ts_open[f],
                                                INPUT_FEATURES[f] * " (norm)",
                                            )
                                        end

                                        
                                        plot_title = @sprintf(
                                            "%s (OPEN)",
                                            INPUT_FEATURES[f],
                                        )
                                        title!(
                                            p_ts_open[f],
                                            plot_title,
                                        )

                                        plot!(p_ts_open[f];
                                            legend=:bottomright,
                                            legendfontsize=legend_fs,
                                            background_color_legend=legend_bg)

                                    end

                                    xlabel!(p_ts_open[feature_size], time_axis_label())

                                    idx_meta2 = feature_size + 1
                                    plot!(p_ts_open[idx_meta2], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)
                                    annotate!(p_ts_open[idx_meta2], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("OPEN-loop horizon=%d", ahead), 14, :center))
                                end

                                ts_closed_path = nothing

                                if RUN_CLOSED_LOOP
                                    p_ts_closed = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    for f in 1:feature_size
                                        y_true_f = @view fut_plot[f, :]
                                        y_closed_f = fill(NaN, N_fut)

                                        fpC = best_fp_closed_per_feature[f]
                                        if fpC != 0
                                            y_closed_f .= @view closed_pred_zscore_all[fpC][f, :]
                                        end

                                        plot!(p_ts_closed[f], t_axis[mask], y_true_f[mask];
                                            lw=2,
                                            seriescolor=c_true,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label="True")

                                        plot!(p_ts_closed[f], t_axis[mask], y_closed_f[mask];
                                            lw=2,
                                            seriescolor=c_closed,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label = fpC == 0 ? "Closed" :
                                                @sprintf(
                                                    "Closed b%02d, best %.1e, mean %.3e",
                                                    fpC,
                                                    best_closed_score[f],
                                                    avg_closed_rmse[f],
                                                ))

                                        ylabel!(p_ts_closed[f], INPUT_FEATURES[f] * " (norm)")
                                        title!(p_ts_closed[f], @sprintf("%s  (CLOSED)", INPUT_FEATURES[f]))

                                        plot!(p_ts_closed[f];
                                            legend=:bottomright,
                                            legendfontsize=legend_fs,
                                            background_color_legend=legend_bg)
                                    end

                                    xlabel!(p_ts_closed[feature_size], time_axis_label())

                                    idx_meta_closed = feature_size + 1
                                    plot!(p_ts_closed[idx_meta_closed], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)

                                    annotate!(p_ts_closed[idx_meta_closed], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("CLOSED-loop horizon=%d", ahead), 14, :center))

                                    fname_ts_closed = "timeseries_" * closed_loop_mode_name() * "_BESTPERFEATURE_" * tag * ".png"
                                    ts_closed_path = joinpath(FIGURES_DIR, fname_ts_closed)
                                end
                                # ---------------- FIGURE B2: OPEN Time-series WITH CTX+FUT raw + selected extrema ----------------
                                ts_open_ctx_path = nothing

                                if !RUN_CLOSED_LOOP
                                    p_ts_open_ctx = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    for f in 1:feature_size
                                        fpO = best_fp_open_per_feature[f]

                                        y_true_f = @view fut_plot[f, :]
                                        y_open_f = fill(NaN, N_fut)

                                        if fpO != 0
                                            y_open_f .= @view open_pred_zscore_all[fpO][f, :]
                                        end

                                        if MAPLIKE
                                            t_raw_all = vcat(t_ctx_raw, t_fut_raw)
                                            x_raw_all_norm = vcat(vec(ctx_zscore_raw[f, :]), vec(fut_zscore_raw[f, :]))

                                            plot!(p_ts_open_ctx[f], t_raw_all, x_raw_all_norm;
                                                lw=2, seriescolor=:black, seriesalpha=0.18,
                                                label="Actual (CTX+FUT)")

                                            x_ctx_norm = vec(ctx_zscore_raw[f, :])
                                            x_fut_norm = vec(fut_zscore_raw[f, :])

                                            idx_ctx = extrema_indices_from_normalized(Float64.(x_ctx_norm); which=MAPLIKE_WHICH, feature_name=INPUT_FEATURES[f])
                                            idx_fut = extrema_indices_from_normalized(Float64.(x_fut_norm); which=MAPLIKE_WHICH, feature_name=INPUT_FEATURES[f])

                                            plot!(p_ts_open_ctx[f], t_ctx_raw[idx_ctx], x_ctx_norm[idx_ctx];
                                                lw=0, marker=:diamond, markersize=6,
                                                seriescolor=:green, label="Selected extrema (CTX used)")

                                            plot!(p_ts_open_ctx[f], t_fut_raw[idx_fut], x_fut_norm[idx_fut];
                                                lw=0, marker=:utriangle, markersize=7,
                                                seriescolor=:blue, label="Selected extrema (FUT used)")

                                            if MAPLIKE_TARGET === :time
                                                y_open_t = fill(NaN, N_fut)
                                                if fpO != 0
                                                    y_open_t .= @view open_pred_timepos_all[fpO][f, :]
                                                end

                                                for tt in t_axis
                                                    vline!(p_ts_open_ctx[f], [tt], color=:blue, alpha=0.25, label=false)
                                                end
                                                for tt in y_open_t[isfinite.(y_open_t)]
                                                    vline!(p_ts_open_ctx[f], [tt], color=:orange, alpha=0.25, label=false)
                                                end
                                            else
                                                plot!(p_ts_open_ctx[f], t_axis, y_true_f;
                                                    lw=2, seriescolor=:blue, marker=:circle, markersize=2,
                                                    label="True (FUT extrema)")

                                                plot!(p_ts_open_ctx[f], t_axis, y_open_f;
                                                    lw=2, seriescolor=:orange, marker=:circle, markersize=2,
                                                    label="Open")
                                            end

                                            xlims!(
                                                p_ts_open_ctx[f],
                                                (ACTIVE_T_CTX[1], ACTIVE_T_FUT[2]),
                                            )
                                        else
                                            plot!(p_ts_open_ctx[f], t_axis, y_true_f; lw=2, label="True")
                                            plot!(p_ts_open_ctx[f], t_axis, y_open_f; lw=2, label="Open")
                                        end

                                        if MAPLIKE && MAPLIKE_TARGET === :time
                                            ylabel!(p_ts_open_ctx[f], INPUT_FEATURES[f] * " raw signal / extrema times")
                                        else
                                            ylabel!(p_ts_open_ctx[f], INPUT_FEATURES[f] * " (norm)")
                                        end
                                        title!(p_ts_open_ctx[f], @sprintf("%s  (OPEN+CTX view, Horizon=%d)", INPUT_FEATURES[f], ahead))
                                        plot!(p_ts_open_ctx[f]; legend=false)
                                    end

                                    xlabel!(p_ts_open_ctx[feature_size], time_axis_label())

                                    idx_meta2b = feature_size + 1
                                    plot!(p_ts_open_ctx[idx_meta2b], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)
                                    annotate!(p_ts_open_ctx[idx_meta2b], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("OPEN-loop horizon=%d (CTX+FUT view)", ahead), 14, :center))
                                end        
                                # ---------------- FIGURE B2-CLOSED: CLOSED Time-series WITH CTX+FUT raw + selected extrema ----------------
                                ts_closed_ctx_path = nothing

                                if RUN_CLOSED_LOOP
                                    p_ts_closed_ctx = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    for f in 1:feature_size
                                        fpC = best_fp_closed_per_feature[f]

                                        y_true_f = @view fut_plot[f, :]
                                        y_closed_f = fill(NaN, N_fut)

                                        if fpC != 0
                                            y_closed_f .= @view closed_pred_zscore_all[fpC][f, :]
                                        end

                                        if MAPLIKE
                                            t_raw_all = vcat(t_ctx_raw, t_fut_raw)
                                            x_raw_all_norm = vcat(vec(ctx_zscore_raw[f, :]), vec(fut_zscore_raw[f, :]))

                                            plot!(p_ts_closed_ctx[f], t_raw_all, x_raw_all_norm;
                                                lw=2,
                                                seriescolor=:black,
                                                seriesalpha=0.18,
                                                label="Actual (CTX+FUT)")

                                            x_ctx_norm = vec(ctx_zscore_raw[f, :])
                                            x_fut_norm = vec(fut_zscore_raw[f, :])

                                            idx_ctx = extrema_indices_from_normalized(
                                                Float64.(x_ctx_norm);
                                                which=MAPLIKE_WHICH,
                                                feature_name=INPUT_FEATURES[f],
                                            )

                                            idx_fut = extrema_indices_from_normalized(
                                                Float64.(x_fut_norm);
                                                which=MAPLIKE_WHICH,
                                                feature_name=INPUT_FEATURES[f],
                                            )

                                            plot!(p_ts_closed_ctx[f], t_ctx_raw[idx_ctx], x_ctx_norm[idx_ctx];
                                                lw=0,
                                                marker=:diamond,
                                                markersize=6,
                                                seriescolor=:green,
                                                label="Selected extrema (CTX used)")

                                            plot!(p_ts_closed_ctx[f], t_fut_raw[idx_fut], x_fut_norm[idx_fut];
                                                lw=0,
                                                marker=:utriangle,
                                                markersize=7,
                                                seriescolor=:blue,
                                                label="Selected extrema (FUT used)")

                                            if MAPLIKE_TARGET === :time
                                                # You do not currently compute closed_pred_timepos_all.
                                                # So for time-target maplike mode, only draw the true extrema times.
                                                for tt in t_axis
                                                    vline!(p_ts_closed_ctx[f], [tt], color=:blue, alpha=0.25, label=false)
                                                end
                                            else
                                                plot!(p_ts_closed_ctx[f], t_axis, y_true_f;
                                                    lw=2,
                                                    seriescolor=:blue,
                                                    marker=:circle,
                                                    markersize=2,
                                                    label="True (FUT extrema)")

                                                plot!(p_ts_closed_ctx[f], t_axis, y_closed_f;
                                                    lw=2,
                                                    seriescolor=:green,
                                                    marker=:circle,
                                                    markersize=2,
                                                    label="Closed")
                                            end

                                            xlims!(
                                                p_ts_closed_ctx[f],
                                                (ACTIVE_T_CTX[1], ACTIVE_T_FUT[2]),
                                            )
                                        else
                                            plot!(p_ts_closed_ctx[f], t_axis, y_true_f;
                                                lw=2,
                                                seriescolor=:blue,
                                                label="True")

                                            plot!(p_ts_closed_ctx[f], t_axis, y_closed_f;
                                                lw=2,
                                                seriescolor=:green,
                                                label="Closed")
                                        end

                                        if MAPLIKE && MAPLIKE_TARGET === :time
                                            ylabel!(p_ts_closed_ctx[f], INPUT_FEATURES[f] * " raw signal / extrema times")
                                        else
                                            ylabel!(p_ts_closed_ctx[f], INPUT_FEATURES[f] * " (norm)")
                                        end

                                        title!(p_ts_closed_ctx[f], @sprintf("%s  (CLOSED+CTX view, Horizon=%d)", INPUT_FEATURES[f], ahead))
                                        plot!(p_ts_closed_ctx[f]; legend=false)
                                    end

                                    xlabel!(p_ts_closed_ctx[feature_size], time_axis_label())

                                    idx_meta2b_closed = feature_size + 1
                                    plot!(p_ts_closed_ctx[idx_meta2b_closed], [0, 1], [0, 1];
                                        legend=false,
                                        framestyle=:none,
                                        xticks=false,
                                        yticks=false,
                                        linealpha=0)

                                    annotate!(p_ts_closed_ctx[idx_meta2b_closed], 0.5, 0.5,
                                        text(meta_str_common * "\n" * @sprintf("CLOSED-loop horizon=%d (CTX+FUT view)", ahead), 14, :center))
                                end
                                # ---------------- FIGURE C: OPEN Time-series (ONE best batch over ALL features) ----------------
                                ts_open_together_path = nothing

                                if !RUN_CLOSED_LOOP
                                    p_ts_open_together = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    fpT = best_fp_open_together
                                    # --- Restrict time window for Figure C ---

                                    for f in 1:feature_size
                                        if MAPLIKE && MAPLIKE_TARGET === :time
                                            y_true_f = t_axis
                                            y_open_f = fill(NaN, N_fut)

                                            if fpT != 0
                                                y_open_f .= @view open_pred_timepos_all[fpT][f, :]
                                            end

                                            plot!(p_ts_open_together[f], t_axis[mask], y_true_f[mask];
                                                lw=2, seriescolor=:blue, marker=:circle, markerstrokecolor=:auto, markersize=2,
                                                label="True extrema time")

                                            plot!(p_ts_open_together[f], t_axis[mask], y_open_f[mask];
                                                lw=2, seriescolor=:orange, marker=:circle, markerstrokecolor=:auto, markersize=2,
                                                label=fpT == 0 ? "Open (no batch)" :
                                                    @sprintf("Predicted extrema time  b%02d\nRMSE(this feature all batches) %.2e\nRMSE(all features all batches) %.2e",
                                                        fpT, avg_open_rmse[f], rmse_all_features_all_batches))

                                            
                                            
                                            ylabel!(p_ts_open_together[f], INPUT_FEATURES[f] * " extrema time ($TIME_UNIT)")
                                        else
                                            y_true_f = @view fut_plot[f, :]
                                            y_open_f = fill(NaN, N_fut)

                                            if fpT != 0
                                                y_open_f .= @view open_pred_zscore_all[fpT][f, :]
                                            end

                                            plot!(p_ts_open_together[f], t_axis[mask], y_true_f[mask];
                                                lw=2, seriescolor=:blue, marker=:circle, markerstrokecolor=:auto, markersize=2,
                                                label="True")

                                            plot!(p_ts_open_together[f], t_axis[mask], y_open_f[mask];
                                                lw=2, seriescolor=:orange, marker=:circle, markerstrokecolor=:auto, markersize=2,
                                                label=fpT == 0 ? "Open (no batch)" :
                                                    @sprintf("Open  b%02d (together shown)\nRMSE(this feature all batches) %.2e\nRMSE(all features all batches) %.2e",
                                                        fpT, avg_open_rmse[f], rmse_all_features_all_batches))


                                            ylabel!(p_ts_open_together[f], INPUT_FEATURES[f] * " (norm)")
                                        end
                                        title!(p_ts_open_together[f], @sprintf("%s  (OPEN, Horizon=%d)", INPUT_FEATURES[f], ahead))
                                        plot!(p_ts_open_together[f]; legend=:outerright, legendfontsize=legend_fs)
                                    end

                                    xlabel!(p_ts_open_together[feature_size], time_axis_label())

                                    idx_meta3 = feature_size + 1
                                    plot!(p_ts_open_together[idx_meta3], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)
                                    annotate!(p_ts_open_together[idx_meta3], 0.5, 0.5,
                                        text(meta_str_common * "\n" *
                                            @sprintf("OPEN-loop horizon=%d\nBEST BATCH OVER ALL FEATURES: b%02d  RMSE(all)=%.2e",
                                                ahead, fpT, best_open_score_together),
                                            14, :center))
                                end




                                ts_closed_together_path = nothing

                                if RUN_CLOSED_LOOP
                                    p_ts_closed_together = plot(
                                        layout=(feature_size + 1, 1),
                                        size=(900, 250 * (feature_size + 1)),
                                        left_margin=20mm,
                                        right_margin=20mm,
                                        titlefont=14,
                                        guidefont=12,
                                        tickfont=11,
                                    )

                                    fpCT = best_fp_closed_together

                                    for f in 1:feature_size
                                        y_true_f = @view fut_plot[f, :]
                                        y_closed_f = fill(NaN, N_fut)

                                        if fpCT != 0
                                            y_closed_f .= @view closed_pred_zscore_all[fpCT][f, :]
                                        end

                                        plot!(p_ts_closed_together[f], t_axis[mask], y_true_f[mask];
                                            lw=2,
                                            seriescolor=:blue,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label="True")

                                        plot!(p_ts_closed_together[f], t_axis[mask], y_closed_f[mask];
                                            lw=2,
                                            seriescolor=:green,
                                            marker=:circle,
                                            markerstrokecolor=:auto,
                                            markersize=2,
                                            label=fpCT == 0 ? "Closed (no batch)" :
                                                @sprintf("Closed  b%02d (together shown)\nRMSE(all features all batches) %.2e",
                                                    fpCT, rmse_closed_all_features_all_batches))

                                        ylabel!(p_ts_closed_together[f], INPUT_FEATURES[f] * " (norm)")
                                        title!(p_ts_closed_together[f], @sprintf("%s  (CLOSED, Horizon=%d)", INPUT_FEATURES[f], ahead))

                                        plot!(p_ts_closed_together[f]; legend=:outerright, legendfontsize=legend_fs)
                                    end

                                    xlabel!(p_ts_closed_together[feature_size], time_axis_label())

                                    idx_meta_closed_together = feature_size + 1
                                    plot!(p_ts_closed_together[idx_meta_closed_together], [0, 1], [0, 1];
                                        legend=false, framestyle=:none, xticks=false, yticks=false, linealpha=0)

                                    annotate!(p_ts_closed_together[idx_meta_closed_together], 0.5, 0.5,
                                        text(meta_str_common * "\n" *
                                            @sprintf("CLOSED-loop horizon=%d\nBEST CLOSED BATCH OVER ALL FEATURES: b%02d  RMSE(closed all)=%.2e",
                                                ahead, fpCT, best_closed_score_together),
                                            14, :center))

                                    fname_ts_closed_together = "timeseries_" * closed_loop_mode_name() * "_BESTBATCH_ALLFEATURES_" * tag * "_together.png"
                                    ts_closed_together_path = joinpath(FIGURES_DIR, fname_ts_closed_together)
                                end




                                if !RUN_CLOSED_LOOP
                                    fname_ts_open = "timeseries_OPEN_BESTPERFEATURE_" * tag * ".png"
                                    ts_open_path = joinpath(FIGURES_DIR, fname_ts_open)

                                    fname_ts_open_together = "timeseries_OPEN_BESTBATCH_ALLFEATURES_" * tag * "_together.png"
                                    ts_open_together_path = joinpath(FIGURES_DIR, fname_ts_open_together)

                                    fname_ts_open_ctx = "timeseries_OPEN_BESTPERFEATURE_CTXPLUSFUT_" * tag * ".png"
                                    ts_open_ctx_path = joinpath(FIGURES_DIR, fname_ts_open_ctx)

                                    # savefig(p_ts_open, ts_open_path)
                                    @info "Saved OPEN time-series figure to $ts_open_path"

                                    # savefig(p_ts_open_together, ts_open_together_path)
                                    @info "Saved OPEN time-series (best batch over all features) figure to $ts_open_together_path"

                                    # savefig(p_ts_open_ctx, ts_open_ctx_path)
                                    @info "Saved OPEN time-series figure (CTX+FUT view) to $ts_open_ctx_path"
                                else
                                    fname_ts_closed = "timeseries_" * closed_loop_mode_name() * "_BESTPERFEATURE_" * tag * ".png"
                                    ts_closed_path = joinpath(FIGURES_DIR, fname_ts_closed)

                                    fname_ts_closed_together = "timeseries_" * closed_loop_mode_name() * "_BESTBATCH_ALLFEATURES_" * tag * "_together.png"
                                    ts_closed_together_path = joinpath(FIGURES_DIR, fname_ts_closed_together)

                                    fname_ts_closed_ctx = "timeseries_" * closed_loop_mode_name() * "_BESTPERFEATURE_CTXPLUSFUT_" * tag * ".png"
                                    ts_closed_ctx_path = joinpath(FIGURES_DIR, fname_ts_closed_ctx)

                                    # # savefig(p_ts_closed, ts_closed_path)
                                    # @info "Saved CLOSED time-series figure to $ts_closed_path"

                                    # # savefig(p_ts_closed_together, ts_closed_together_path)
                                    # @info "Saved CLOSED time-series (best batch over all features) figure to $ts_closed_together_path"

                                    # # savefig(p_ts_closed_ctx, ts_closed_ctx_path)
                                    # @info "Saved CLOSED time-series figure (CTX+FUT view) to $ts_closed_ctx_path"
                                end
                            end # MAKE_PREDICTION_FIGURES
                            
                            # Save JLD2
                            out_prefix = RUN_CLOSED_LOOP ? closed_loop_file_prefix() : "open_teacherforced_only_"
                            outpath = joinpath(RESULTS_DIR, out_prefix * tag * ".jld2")

                            jldsave(outpath;
                                HAMILTONIAN_KIND=HAMILTONIAN_KIND,
                                HAM_NAME=HAM_NAME,
                                ahead,
                                INPUT_FEATURES,
                                INPUT_MAPPING_MODE=String(INPUT_MAPPING_MODE),
                                NTIMESTEPS=ntimesteps,
                                NSYSTEM=nsystem_run,
                                DATA_MODE = String(DATA_MODE),
                                TIME_UNIT = TIME_UNIT,
                                T_CTX = ACTIVE_T_CTX,
                                T_FUT = ACTIVE_T_FUT,
                                dt_raw_data = median(diff(vcat(t_ctx_raw, t_fut_raw))),
                                lag_time_spacing = ds * median(diff(vcat(t_ctx_raw, t_fut_raw))),
                                DS=ds, N_BATCHES=N_BATCHES,
                                N_LAYERS=nlayers,
                                N_MEMORY=nmemory,
                                λ=λ,
                                τ=τ,
                                LAG_REP=LAG_REP,
                                REPEAT_FEATURES=REPEAT_FEATURES,
                                APPROX_METHOD=String(APPROX_METHOD),
                                normμ=normμ,
                                normσ=normσ,
                                z_to_angle_offset = data_pack.μang,
                                z_to_angle_scale = data_pack.σang,
                                angle_max = ANGLE_MAX,

                                gap_normμ = hasproperty(data_pack, :μgap) ? data_pack.μgap : nothing,
                                gap_normσ = hasproperty(data_pack, :σgap) ? data_pack.σgap : nothing,
                                fut_angle = fut_angle,
                                fut_target_angle = fut_target,
                                fut_target_zscore = fut_plot,
                                fut_plot=fut_plot,
                                t_axis=t_axis,
                                ctx_zscore_raw = ctx_zscore_raw,
                                fut_zscore_raw = fut_zscore_raw,
                                ctx_zscore_eval = ctx_zscore_eval_orig,
                                fut_zscore_eval = fut_zscore_eval_orig,
                                ctx_angle_raw = data_pack.ctx_angle_raw,
                                fut_angle_raw = data_pack.fut_angle_raw,
                                ctx_norm_raw = ctx_zscore_raw,
                                fut_norm_raw = fut_zscore_raw,
                                t_ctx_raw=t_ctx_raw,
                                t_fut_raw=t_fut_raw,
                                open_pred_angle_all = open_pred_angle_all,
                                closed_pred_angle_all = closed_pred_angle_all,

                                open_pred_zscore_all = open_pred_zscore_all,
                                closed_pred_zscore_all = closed_pred_zscore_all,
                                open_rmse_vs_time = open_rmse_vs_time,
                                # backward-compatible aliases for figure recovery scripts
                                open_pred_norm_all = open_pred_zscore_all,
                                closed_pred_norm_all = closed_pred_zscore_all,
                                best_fp_open_per_feature=best_fp_open_per_feature,
                                best_open_score=best_open_score,
                                rmse_open_path=rmse_open_path,
                                ts_open_path=ts_open_path,
                                ts_open_together_path=ts_open_together_path,
                                ts_open_ctx_path=ts_open_ctx_path,
                                rmse_closed_path=rmse_closed_path,
                                ts_closed_path=ts_closed_path,
                                ts_closed_together_path=ts_closed_together_path,
                                ts_closed_ctx_path=ts_closed_ctx_path,
                                closed_rmse_overall_per_batch=closed_rmse_overall_per_batch,
                                best_fp_closed_together=best_fp_closed_together,
                                best_closed_score_together=best_closed_score_together,
                                rmse_closed_all_features_all_batches=rmse_closed_all_features_all_batches,
                                RUN_CLOSED_LOOP=RUN_CLOSED_LOOP,
                                RUN_BLOCKWISE_CLOSED_LOOP=RUN_BLOCKWISE_CLOSED_LOOP,
                                CLOSED_LOOP_MODE=RUN_CLOSED_LOOP ? closed_loop_mode_name() : "OPEN",
                                closed_rmse_vs_time=closed_rmse_vs_time,
                                best_fp_closed_per_feature=best_fp_closed_per_feature,
                                best_closed_score=best_closed_score,
                                avg_closed_rmse=avg_closed_rmse,
                                closed_rmse_per_batch=closed_rmse_per_batch,  )
                        end
                    end
                end
            end
        end
    end
end

# -------------------- RUN --------------------
run_multi_mstep_prediction()
# ============================================================
# RUN SELECTED DECOHERENCE CONFIGURATIONS
# ============================================================

# run_one_selected_decoherence_job(
#     "4to1",
#     "u1",
# )
# run_selected_decoherence_jobs()
