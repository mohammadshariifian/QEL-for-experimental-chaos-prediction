# =============================================================================
# Time_serial_circuit_4to1_v3.jl
#
# 独立脚本：重算 4to1 若干“最优配置”下 100 个随机 Ham 的测试 RMSE，
# 输出 mean / std（以及逐个 Ham 的 RMSE），供论文表格使用。
#
# 重要：
#   - 本脚本 **只读** result_4to1_v2/... 以读取旧的 avg 做对比；
#   - **不写入/不覆盖** result_4to1_v2 下任何文件；
#   - 所有输出写到 result_4to1_v3/。
#
# 运行方式（必须在项目根目录）：
#   PID=100 NTHREADS=1 julia Time_serial_circuit_4to1_v3.jl
#
# 环境变量：
#   PID         worker 进程数（默认 50）
#   NTHREADS    每个 worker 的 BLAS 线程数（默认 1，越小越可复现）
#   HAM_TYPE    "linear" | "fc"（默认 linear）
#   TAU_MODE    "tau_nqubit" | "tau_1"（默认 tau_nqubit）
#   NQUBIT_MAX  最大比特数（默认 11）
# =============================================================================

push!(LOAD_PATH, "../QuantumCircuits/src", "../VQC/src", "../QuantumReservoirComputing/src")
using Pkg
Pkg.activate(".")
using ProgressBars
using DelimitedFiles
using JLD2
using Distributions
using StatsBase
using CSV, DataFrames
using Distributed
using LinearAlgebra
using Random
using Statistics
using Dates

# =============================================================================
# 环境配置
# =============================================================================

const PID = parse(Int, get(ENV, "PID", "50"))
const NTHREADS = parse(Int, get(ENV, "NTHREADS", "1"))

const HAM_TYPE = get(ENV, "HAM_TYPE", "linear")
const TAU_MODE = get(ENV, "TAU_MODE", "tau_nqubit")
const NQUBIT_MAX = parse(Int, get(ENV, "NQUBIT_MAX", "11"))

nprocs() < PID && addprocs(PID - nprocs() + 1; env=["JULIA_NUM_THREADS" => "$NTHREADS"])
println("The number of processors: ", nprocs(), " threads per worker: ", NTHREADS)

@everywhere begin
    include("src/head.jl")
    using LinearAlgebra
    BLAS.set_num_threads($NTHREADS)
end
BLAS.set_num_threads(NTHREADS)

# =============================================================================
# 常量（与 Time_serial_circuit_4to1_v2.jl 保持一致）
# =============================================================================

const DATA_FILE = joinpath("data", "data_theory.csv")

const VARIABLE_NAMES = Dict(1 => "uC1", 2 => "uC2", 3 => "iL1", 4 => "iL2")
const INPUT_IDS = [1, 2, 3, 4]
const TRAIN_WINDOW = (1563, 65625)
const TEST_WINDOW = (65626, 93750)
const BATCH_SIZE = 100
const BEARING_TERM_WEIGHT = 0

const OUTPUT_DIR = "result_4to1_v3"

function get_variable_name(id)
    haskey(VARIABLE_NAMES, id) || error("Invalid id: $id. Valid ids are 1-4.")
    return VARIABLE_NAMES[id]
end

# 需要重算的“最优配置”（来自论文表格 Theory 行的 Nsteps / Nhid）
# 注意：nmemory 即表格里的 N_hid
const CONFIGS = [
    (output_id = 1, nsteps = 1, nmemory = 7, past_interval = 21),  # uC1
    (output_id = 2, nsteps = 1, nmemory = 7, past_interval = 7),   # uC2
    (output_id = 3, nsteps = 1, nmemory = 7, past_interval = 34),  # iL1
    (output_id = 4, nsteps = 2, nmemory = 3, past_interval = 68),  # iL2
]

# =============================================================================
# 分布式计算函数（与 v2 完全一致）
# =============================================================================

@everywhere function Distributed_train_and_predict(nmemory, nsystem, B, P, H, tau, train_input, train_output, test_input)
    encode_cir = encode_circuit(nsystem)
    noise_cir = nothing

    nqubit = size(train_input, 1) + nmemory
    U = nqubit <= 14 ? convert.(ComplexF32, exp(-im * tau * Matrix(matrix(H)))) : nothing

    train_y = reshape(train_output, size(train_output, 1), size(train_output, 3))
    Lb = length(B)
    W, train_err, train_pred = train_H(train_input, train_y, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    Ws_single = [W]

    signal_test = Quantum_Reservoir_Serial_arrangement_H(
        test_input, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    test_pred = Ws_single[1] * signal_test

    return reshape(W, Lb, 1), [train_err], train_pred, test_pred
end

# =============================================================================
# 滑动窗口（与 v2 完全一致）
# =============================================================================

function build_windows_multi_concat(input_data_vecs, output_data_vec, train_range, test_range, nsteps, horizon=0, past_interval=0;
                                    target_vec=nothing)
    n_vars = length(input_data_vecs)
    input_span = past_interval == 0 ? nsteps : nsteps * past_interval
    tgt = target_vec === nothing ? output_data_vec : target_vec

    train_input_list = []
    train_output_list = []
    train_target_indices = []
    for t in max(train_range[1], input_span + 1):(train_range[2] - horizon)
        if past_interval == 0
            windows = [input_data_vecs[v][t-nsteps:t-1] for v in 1:n_vars]
        else
            windows = [[input_data_vecs[v][t - k * past_interval] for k in nsteps:-1:1] for v in 1:n_vars]
        end
        push!(train_input_list, vcat(windows...))
        push!(train_output_list, tgt[t + horizon])
        push!(train_target_indices, t + horizon)
    end

    test_input_list = []
    test_output_list = []
    test_target_indices = []
    for t in max(test_range[1], input_span + 1):(test_range[2] - horizon)
        if past_interval == 0
            windows = [input_data_vecs[v][t-nsteps:t-1] for v in 1:n_vars]
        else
            windows = [[input_data_vecs[v][t - k * past_interval] for k in nsteps:-1:1] for v in 1:n_vars]
        end
        push!(test_input_list, vcat(windows...))
        push!(test_output_list, tgt[t + horizon])
        push!(test_target_indices, t + horizon)
    end

    N_train = length(train_output_list)
    N_test = length(test_output_list)
    @assert N_train > 0 "训练样本数为0"
    @assert N_test > 0 "测试样本数为0"

    feat_dim = nsteps * n_vars
    train_input = zeros(feat_dim, 1, N_train)
    train_output = zeros(1, 1, N_train)
    for i in 1:N_train
        train_input[:, 1, i] = train_input_list[i]
        train_output[1, 1, i] = train_output_list[i]
    end
    test_input = zeros(feat_dim, 1, N_test)
    test_output = zeros(1, 1, N_test)
    for i in 1:N_test
        test_input[:, 1, i] = test_input_list[i]
        test_output[1, 1, i] = test_output_list[i]
    end

    return train_input, train_output, test_input, test_output, test_target_indices, train_target_indices
end

# =============================================================================
# 量子电路生成（与 v2 完全一致）
# =============================================================================

function generate_quantum_circuits(nqubit, batch_size, evolution_time;
                                   ham_type="linear", seed=nothing)
    seed !== nothing && Random.seed!(seed)
    Hs = []
    npairs = ham_type == "fc" ? nqubit * (nqubit - 1) ÷ 2 : nqubit - 1
    for i in 1:batch_size
        Jx = rand(npairs); Jy = rand(npairs); Jz = rand(npairs)
        m = max(maximum(Jx), maximum(Jy), maximum(Jz))
        if m > 0; Jx ./= m; Jy ./= m; Jz ./= m; end
        if ham_type == "fc"
            QrH = Ham_fc(nqubit, Jx, Jy, Jz)
        else
            QrH = Ham_XYZ_nn(nqubit, Jx, Jy, Jz)
        end
        push!(Hs, QrH)
    end
    return Hs
end

# =============================================================================
# 训练与预测核心（与 v2 一致，额外返回 batch_rmses）
# =============================================================================

function train_and_predict_multi(input_data_vecs, output_data_vec, train_window, test_window,
                                 nmemory, nsteps, B, P, Hs, tau, batch_size,
                                 data_min, data_max, horizon=0, past_interval=0)
    train_input, train_output, test_input, test_output, test_target_indices, train_target_indices =
        build_windows_multi_concat(input_data_vecs, output_data_vec, train_window, test_window, nsteps, horizon, past_interval)

    nsystem = size(train_input, 1)
    LB = length(B)
    Ws = zeros(4, LB, 1, batch_size)
    EEs = zeros(4, 1, batch_size)

    F = []
    for fp in ProgressBar(1:batch_size)
        push!(F, Distributed.@spawn Distributed_train_and_predict(
            nmemory, nsystem, B, P, Hs[fp], tau,
            train_input, train_output, test_input
        ))
    end

    batch_rmses = zeros(batch_size)
    best_index = 1
    best_ee = Inf
    best_train_pred = zeros(1, 1)
    best_test_pred = zeros(1, 1)
    for fp in 1:batch_size
        W, train_err, train_pred, test_pred = fetch(F[fp])
        Ws[1, :, :, fp] = W
        EEs[1, :, fp] = train_err
        n_compare = min(size(test_output, 3), size(test_pred, 2))
        pred_z = vec(test_pred)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
        actual_z = vec(test_output)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
        batch_rmses[fp] = sqrt(mean((actual_z .- pred_z).^2))
        if train_err[1] < best_ee
            best_ee = train_err[1]
            best_index = fp
            best_train_pred = train_pred
            best_test_pred = test_pred
        end
    end
    avg_variance = mean(batch_rmses)

    output_train = best_train_pred
    output_test = best_test_pred

    n_compare = min(size(test_output, 3), size(output_test, 2))
    pred_z = vec(output_test)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
    actual_z = vec(test_output)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
    variance = sqrt(mean((actual_z .- pred_z).^2))

    return output_train, output_test, variance, Ws, EEs, test_target_indices, avg_variance, train_target_indices, batch_rmses
end

# =============================================================================
# 数据准备（与 v2 main_multi 完全一致）
# =============================================================================

function prepare_data(output_id)
    raw_Data = readdlm(DATA_FILE, ',')

    train_raw = raw_Data[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    Data = (raw_Data .- mu) ./ sigma

    all_input_data = vcat([Data[id, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]] for id in INPUT_IDS]...)
    data_min = minimum(all_input_data)
    data_max = maximum(all_input_data)

    for id in INPUT_IDS
        Data[id, :] = (Data[id, :] .- data_min) ./ (data_max - data_min) .* π
    end

    input_vecs = [Data[id, :] for id in INPUT_IDS]
    output_vec = Data[output_id, :]
    return input_vecs, output_vec, data_min, data_max
end

# =============================================================================
# 读取旧的 avg（仅用于对比）
# =============================================================================

function read_old_values(output_name, past_interval, nsteps, nmemory)
    tau_label = TAU_MODE == "tau_1" ? "tau1" : "taun"
    config_label = "$(HAM_TYPE)_$(tau_label)"
    path = joinpath("result_4to1_v2", config_label, "$(output_name)_$(past_interval)", "results.jld2")
    isfile(path) || return (NaN, NaN)
    res = load(path, "results")
    haskey(res, (nsteps, nmemory)) || return (NaN, NaN)
    best, avg = res[(nsteps, nmemory)]
    return (best, avg)
end

# =============================================================================
# 主流程
# =============================================================================

function main()
    println("HAM_TYPE=$HAM_TYPE, TAU_MODE=$TAU_MODE, NQUBIT_MAX=$NQUBIT_MAX")
    println("Configs to recompute:")
    for c in CONFIGS
        println("  ", get_variable_name(c.output_id), " nsteps=", c.nsteps,
                " nmemory=", c.nmemory, " past_interval=", c.past_interval,
                " nqubit=", c.nsteps * length(INPUT_IDS) + c.nmemory)
    end

    isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)

    stds = Dict{String, Any}()
    rows = NamedTuple[]

    for cfg in CONFIGS
        output_name = get_variable_name(cfg.output_id)
        nqubit = cfg.nsteps * length(INPUT_IDS) + cfg.nmemory
        if nqubit > NQUBIT_MAX
            println("[skip] $output_name nqubit=$nqubit > NQUBIT_MAX=$NQUBIT_MAX")
            continue
        end

        label = "$(output_name)_S$(cfg.nsteps)_M$(cfg.nmemory)"
        println("\n[$(now())] === $label (nsteps=$(cfg.nsteps), nmemory=$(cfg.nmemory), past_interval=$(cfg.past_interval)) ===")

        input_vecs, output_vec, data_min, data_max = prepare_data(cfg.output_id)

        tau = TAU_MODE == "tau_1" ? 1.0 : Float64(nqubit)
        B = vcat([QubitsTerm(i => "X") for i in 1:nqubit],
                 [QubitsTerm(i => "Y") for i in 1:nqubit],
                 [QubitsTerm(i => "Z") for i in 1:nqubit])
        P = BEARING_TERM_WEIGHT
        Hs = generate_quantum_circuits(nqubit, BATCH_SIZE, tau; ham_type=HAM_TYPE, seed=42)

        local batch_rmses
        try
            res = train_and_predict_multi(
                input_vecs, output_vec, TRAIN_WINDOW, TEST_WINDOW,
                cfg.nmemory, cfg.nsteps, B, P, Hs, tau, BATCH_SIZE,
                data_min, data_max, 0, cfg.past_interval
            )
            batch_rmses = res[end]
        catch e
            println("  ✗ 失败: ", e)
            continue
        end

        m = mean(batch_rmses)
        s = std(batch_rmses)
        old_best, old_avg = read_old_values(output_name, cfg.past_interval, cfg.nsteps, cfg.nmemory)

        println("  recomputed mean = $m")
        println("  recomputed std  = $s")
        println("  recomputed min  = $(minimum(batch_rmses))  (best-Ham, v2 best=$old_best)")
        println("  old v2 avg      = $old_avg   delta(mean-old_avg) = $(m - old_avg)")

        stds[label] = Dict(
            "variable" => output_name,
            "nsteps" => cfg.nsteps,
            "nmemory" => cfg.nmemory,
            "past_interval" => cfg.past_interval,
            "mean" => m,
            "std" => s,
            "min" => minimum(batch_rmses),
            "old_avg" => old_avg,
            "old_best" => old_best,
            "batch_rmses" => batch_rmses,
        )

        push!(rows, (
            var = output_name,
            nsteps = cfg.nsteps,
            nmemory = cfg.nmemory,
            past_interval = cfg.past_interval,
            mean = m,
            std = s,
            min = minimum(batch_rmses),
            old_avg = old_avg,
            delta = m - old_avg,
        ))

        save(joinpath(OUTPUT_DIR, "stds.jld2"), "stds", stds)
        CSV.write(joinpath(OUTPUT_DIR, "stds_summary.csv"), DataFrame(rows))
    end

    println("\n[$(now())] Done. Outputs in $(OUTPUT_DIR)/ :")
    println("  - $(joinpath(OUTPUT_DIR, "stds.jld2"))        (mean/std/batch_rmses per config)")
    println("  - $(joinpath(OUTPUT_DIR, "stds_summary.csv")) (summary table)")
    println("\n最简结论（论文表格用）：")
    for r in rows
        println(rpad(r.var, 5), " (Nsteps=", r.nsteps, ", Nhid=", r.nmemory, ")  ",
                round(r.mean, digits=4), " ± ", round(r.std, digits=4),
                "   (旧 avg=", round(r.old_avg, digits=4), ", Δ=", round(r.delta, digits=6), ")")
    end
end

main()
