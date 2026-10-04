push!(LOAD_PATH,"../QuantumCircuits/src","../VQC/src","../QuantumReservoirComputing/src")
using Pkg
Pkg.activate(".")
using ProgressBars
using DelimitedFiles
using Plots
using Plots.PlotMeasures
using JLD2
using Distributions
using StatsBase
using CSV, DataFrames
using Distributed
using LinearAlgebra
using Random

# 命令行参数解析（向后兼容默认值）
ham_type = length(ARGS) >= 1 ? ARGS[1] : "linear"
tau_mode = length(ARGS) >= 2 ? ARGS[2] : "tau_nqubit"
nqubit_max = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 11
var_id = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0  # 0=all, 1-4=single variable
function _parse_range(s::AbstractString)
    if contains(s, ":")
        a, b = split(s, ":")
        parse(Int, a):parse(Int, b)
    else
        v = parse(Int, s); v:v
    end
end
nsteps_val = length(ARGS) >= 5 ? _parse_range(ARGS[5]) : 1:3
nmemory_val = length(ARGS) >= 6 ? _parse_range(ARGS[6]) : 1:10
batch_size_val = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 100
nqubit_min = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 0  # 0=compute all; >0=only nqubit ≥ nqubit_min
past_interval_override = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 0  # 0=use default per-variable interval; >0=force this interval

# =============================================================================
# 配置参数
# =============================================================================

# 分布式计算配置
const PID = parse(Int, get(ENV, "PID", "50"))

# 数据配置
const DATA_FILE = joinpath("data", "data_theory.csv")

# =============================================================================
# 变量名映射规则
# =============================================================================

const VARIABLE_NAMES = Dict(
    1 => "uC1",
    2 => "uC2",
    3 => "iL1",
    4 => "iL2"
)

function get_variable_name(id)
    if !haskey(VARIABLE_NAMES, id)
        error("Invalid id: $id. Valid ids are 1-4.")
    end
    return VARIABLE_NAMES[id]
end

# 输入变量 (全部4个): uC1, uC2, iL1, iL2
const INPUT_IDS = [1, 2, 3, 4]

# 训练/测试时间窗口配置
const TRAIN_WINDOW = (1563, 65625)
const TEST_WINDOW = (65626, 93750)

# 量子系统参数
const BATCH_SIZE = 100
const BEARING_TERM_WEIGHT = 0

# =============================================================================
# 环境初始化
# =============================================================================

const NTHREADS = parse(Int, get(ENV, "NTHREADS", "1"))
nprocs() < PID && addprocs(PID - nprocs() + 1; env=["JULIA_NUM_THREADS"=>"$NTHREADS"])
println("The number of processors: ", nprocs(), " threads per worker: ", NTHREADS)

@everywhere begin
    include("src/head.jl")
    using LinearAlgebra
    BLAS.set_num_threads($NTHREADS)
end
BLAS.set_num_threads(NTHREADS)

# =============================================================================
# 分布式计算函数
# =============================================================================

@everywhere function Distributed_train_and_predict(nmemory, nsystem, B, P, H, tau, train_input, train_output, test_input)
    encode_cir = encode_circuit(nsystem)
    noise_cir = nothing

    # 构建 U 矩阵 (只构建一次)
    nqubit = size(train_input, 1) + nmemory
    U = nqubit <= 14 ? convert.(ComplexF32, exp(-im * tau * Matrix(matrix(H)))) : nothing

    # 训练
    train_y = reshape(train_output, size(train_output, 1), size(train_output, 3))
    Lb = length(B)
    W, train_err, train_pred = train_H(train_input, train_y, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    Ws_single = [W]

    # 测试集预测 (批量)
    signal_test = Quantum_Reservoir_Serial_arrangement_H(
        test_input, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    test_pred = Ws_single[1] * signal_test

    return reshape(W, Lb, 1), [train_err], train_pred, test_pred
end

# =============================================================================
# 数据处理工具函数
# =============================================================================

"""
多变量滑动窗口构建（Concat 布局）

输入 shape: (nsteps * n_vars, 1, N)
- 每个变量的 nsteps 个值拼接成长向量
- 所有特征在一个时间步 (ys=1) 内编码

# 参数
- `input_data_vecs`: Vector of Vector — 每个输入变量的标准化数据
- `output_data_vec`: 输出变量的标准化数据
- `train_range`, `test_range`: 时间范围
- `nsteps`: 输入步数
- `horizon`: 预测步数偏移

# 返回
- `train_input`: (nsteps * n_vars, 1, N_train)
- `train_output`: (1, 1, N_train)
- `test_input`: (nsteps * n_vars, 1, N_test)
- `test_output`: (1, 1, N_test)
- `test_target_indices`
"""
function build_windows_multi_concat(input_data_vecs, output_data_vec, train_range, test_range, nsteps, horizon=0, past_interval=0;
                                    target_vec=nothing)
    n_vars = length(input_data_vecs)
    # past_interval=0: 连续取值（步长=1）；否则按 past_interval 间隔采样
    input_span = past_interval == 0 ? nsteps : nsteps * past_interval
    tgt = target_vec === nothing ? output_data_vec : target_vec

    # 训练样本
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

    # 测试样本
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
# 量子电路配置函数
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
# 训练与预测核心流程
# =============================================================================

"""
执行单个参数组合的训练和预测（Concat 布局）
- 数据已在 [0,π] 空间（main_multi 中完成归一化）
- 预测在 [0,π] 空间，RMSE 反变换到 z-score 空间计算
"""
function train_and_predict_multi(input_data_vecs, output_data_vec, train_window, test_window,
                                  nmemory, nsteps, B, P, Hs, tau, batch_size,
                                  data_min, data_max, horizon=0, past_interval=0)

    # 构建训练和测试窗口（数据已在 [0,π] 空间）
    train_input, train_output, test_input, test_output, test_target_indices, train_target_indices =
        build_windows_multi_concat(input_data_vecs, output_data_vec, train_window, test_window, nsteps, horizon, past_interval)

    nsystem = size(train_input, 1)  # nsteps * n_vars
    LB = length(B)
    Ws = zeros(4, LB, 1, batch_size)
    EEs = zeros(4, 1, batch_size)

    # 分布式训练+预测
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
        # 反变换到 z-score 空间计算 RMSE
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

    # 反变换到 z-score 空间计算 RMSE
    n_compare = min(size(test_output, 3), size(output_test, 2))
    pred_z = vec(output_test)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
    actual_z = vec(test_output)[1:n_compare] ./ π .* (data_max - data_min) .+ data_min
    variance = sqrt(mean((actual_z .- pred_z).^2))

    return output_train, output_test, variance, Ws, EEs, test_target_indices, avg_variance, train_target_indices
end

# =============================================================================
# 主程序
# =============================================================================

function main_multi(output_id; nsteps_range=1:5, nmemory_range=1:5, horizon=0, past_interval=0,
                    ham_type="linear", tau_mode="tau_nqubit", nqubit_max=11, nqubit_min=0)
    output_name = get_variable_name(output_id)
    input_names = [get_variable_name(id) for id in INPUT_IDS]

    println("Processing: input=$(input_names) -> output=$(output_name) [Concat, horizon=$horizon, past_interval=$past_interval]")

    # 读取数据
    raw_Data = readdlm(DATA_FILE, ',')

    # 数据标准化 —— 只用训练集原始数据计算 mu/sigma，避免数据泄露
    train_raw = raw_Data[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    Data = (raw_Data .- mu) ./ sigma

    # 计算全局 min/max（用于 [0,π] 归一化）—— 只用训练集 Z-score 数据
    all_input_data = vcat([Data[id, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]] for id in INPUT_IDS]...)
    data_min = minimum(all_input_data)
    data_max = maximum(all_input_data)

    # 归一化所有数据到 [0,π]
    for id in INPUT_IDS
        Data[id, :] = (Data[id, :] .- data_min) ./ (data_max - data_min) .* π
    end

    # 提取输入和输出数据（已在 [0,π] 空间）
    input_vecs = [Data[id, :] for id in INPUT_IDS]
    output_vec = Data[output_id, :]

    # 输出目录（ham类型+tau+变量名区分）
    tau_label = tau_mode == "tau_1" ? "tau1" : "taun"
    config_label = "$(ham_type)_$(tau_label)"
    var_dir = joinpath("result_4to1_v2", config_label, "$(output_name)_$(past_interval)")
    isdir(var_dir) || mkpath(var_dir)

    # 加载已有的 results 字典
    checkpoint_file = joinpath(var_dir, "results.jld2")
    if isfile(checkpoint_file)
        results = load(checkpoint_file, "results")
        println("Loaded existing results from $(checkpoint_file)")
    else
        results = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
        println("Initialized empty results dict")
    end

    # 遍历参数组合
    for nsteps in nsteps_range
        for nmemory in nmemory_range
            # 量子系统配置 — nsteps * 4 个系统比特 + nmemory 个记忆比特
            nqubit = nsteps * 4 + nmemory
            nqubit > nqubit_max && continue
            nqubit_min > 0 && nqubit < nqubit_min && continue
            println("Processing: nsteps=$nsteps, nmemory=$nmemory")
            tau = tau_mode == "tau_1" ? 1.0 : Float64(nqubit)
            B = vcat([QubitsTerm(i => "X") for i in 1:nqubit], [QubitsTerm(i => "Y") for i in 1:nqubit], [QubitsTerm(i => "Z") for i in 1:nqubit])
            P = BEARING_TERM_WEIGHT

            # 生成量子电路
            Hs = generate_quantum_circuits(nqubit, BATCH_SIZE, tau; ham_type=ham_type, seed=42)

            # 训练和预测
            local output_train, output_test, variance, _, _, test_target_indices, avg_variance, train_target_indices

            try
                output_train, output_test, variance, _, _, test_target_indices, avg_variance, train_target_indices = train_and_predict_multi(
                    input_vecs, output_vec, TRAIN_WINDOW, TEST_WINDOW,
                    nmemory, nsteps, B, P, Hs, tau, BATCH_SIZE,
                    data_min, data_max, horizon, past_interval
                )
            catch e
                println("  ✗ 处理失败: ", e)
                results[(nsteps, nmemory)] = (NaN, NaN)
                continue
            end

            results[(nsteps, nmemory)] = (variance, avg_variance)
            println("  Test RMSE (Best): $variance, (Avg): $avg_variance")

            # 保存预测结果
            prediction_file = joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory).jld2")
            save(prediction_file,
                "output_test", output_test,
                "test_target_indices", test_target_indices,
                "output_train", output_train,
                "train_target_indices", train_target_indices,
                "data_min", data_min,
                "data_max", data_max)

            # 保存 checkpoint
            save(checkpoint_file, "results", results)

        end
    end

    println("Done! Results saved to $(var_dir)/")
end

# =============================================================================
# 运行入口
# =============================================================================

function run_all(id; ham_type="linear", tau_mode="tau_nqubit", nqubit_max=11, nqubit_min=0, past_interval_override=0)
    horizons = [21,7,34,68]
    past_interval = past_interval_override > 0 ? past_interval_override : horizons[id]
    println("Running for id=$id [ham=$ham_type, tau=$tau_mode, nqubit_max=$nqubit_max, nqubit_min=$nqubit_min, past_interval=$(past_interval)]")
    main_multi(id, nsteps_range=nsteps_val, nmemory_range=nmemory_val,
               horizon=0, past_interval=past_interval,
               ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min)
end

if var_id > 0
    run_all(var_id, ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min,
            past_interval_override=past_interval_override)
else
    for id in 1:4
        run_all(id, ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min,
                past_interval_override=past_interval_override)
    end
end


