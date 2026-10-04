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
using Dates

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
nsteps_val = length(ARGS) >= 5 ? _parse_range(ARGS[5]) : 3:3
nmemory_val = length(ARGS) >= 6 ? _parse_range(ARGS[6]) : 3:3
batch_size_val = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 100
nqubit_min = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 0  # 0=compute all nqubit ≤ max; >0=only compute nqubit == nqubit_min
past_interval_override = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : 0  # 0=use PAST_INTERVALS default

# =============================================================================
# 配置参数
# =============================================================================

# 线程配置（通过环境变量 NTHREADS 控制）
const NTHREADS = parse(Int, get(ENV, "NTHREADS", "1"))

# 分布式计算配置
const PID = 50

# 数据配置
const DATA_FILE = joinpath("data", "data_theory.csv")
const OUTPUT_DIR = "result_1to1_v2"

# =============================================================================
# 变量名映射规则
# =============================================================================
# id=1 -> uC1
# id=2 -> uC2
# id=3 -> iL1
# id=4 -> iL2

const VARIABLE_NAMES = Dict(
    1 => "uC1",
    2 => "uC2",
    3 => "iL1",
    4 => "iL2"
)

"""
根据id获取变量名
"""
function get_variable_name(id)
    if !haskey(VARIABLE_NAMES, id)
        error("Invalid id: $id. Valid ids are 1-4.")
    end
    return VARIABLE_NAMES[id]
end

# 训练/测试时间窗口配置
const TRAIN_WINDOW = (1563, 65625)   # 训练时间范围（原始数据索引）
const TEST_WINDOW = (65626, 93750)    # 测试时间范围（原始数据索引）

# 量子系统参数
BATCH_SIZE = batch_size_val     # 批次大小（可通过命令行设置）
const BEARING_TERM_WEIGHT = 0   # 噪声参数

# =============================================================================
# 环境初始化
# =============================================================================

# 启动分布式进程（限制每个子进程使用2个线程）
nprocs() < PID && addprocs(PID - nprocs() + 1; env=["JULIA_NUM_THREADS"=>"$NTHREADS"])
println("The number of processors: ", nprocs(), " threads per worker: ", NTHREADS)

# 配置分布式环境
@everywhere begin
    include("src/head.jl")
    using LinearAlgebra
    BLAS.set_num_threads($NTHREADS)
end
BLAS.set_num_threads(NTHREADS)

# =============================================================================
# 分布式计算函数
# =============================================================================

# 分布式训练+预测合并函数
# - 单次 @spawn 完成训练和测试集预测，避免重复分发
# - 返回: (W, train_err, train_pred, test_pred)
@everywhere function Distributed_train_and_predict(
    nmemory, nsystem, B, P, H, tau, train_input, train_output, test_input)
    t_start = time()
    encode_cir = encode_circuit(nsystem)
    noise_cir = nothing

    # 构建 U 矩阵 (只构建一次, 训练和测试共用)
    nqubit = size(train_input, 1) + nmemory
    U = nqubit <= 14 ? convert.(ComplexF32, exp(-im * tau * Matrix(matrix(H)))) : nothing
    t_after_U = time()

    # 训练
    train_y = reshape(train_output, size(train_output, 1), size(train_output, 3))
    Lb = length(B)
    W, train_err, train_pred = train_H(train_input, train_y, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    Ws_single = [W]
    t_after_train = time()

    # 测试集预测 (批量)
    signal_test = Quantum_Reservoir_Serial_arrangement_H(
        test_input, nmemory, encode_cir, noise_cir, H, tau, B; U=U)
    test_pred = Ws_single[1] * signal_test
    t_after_test = time()

    total = round(t_after_test - t_start, sigdigits=2)
    u_time = round(t_after_U - t_start, sigdigits=2)
    train_time = round(t_after_train - t_after_U, sigdigits=2)
    test_time = round(t_after_test - t_after_train, sigdigits=2)
    println("    [worker] nqubit=$nqubit | U=$(u_time)s train=$(train_time)s test=$(test_time)s total=$(total)s err=$(round(train_err[1],digits=6))")
    flush(stdout)

    return reshape(W, Lb, 1), [train_err], train_pred, test_pred
end

# =============================================================================
# 数据处理工具函数
# =============================================================================

"""
基于连续索引构建训练/测试滑动窗口（所有数据点版本）

# 参数
- `data_vec`: 标准化后的1维数据向量（单个变量）
- `train_range`: 训练时间范围 (start, end)，按原始数据索引
- `test_range`: 测试时间范围 (start, end)，按原始数据索引
- `nsteps`: 输入步数（历史数据点个数）

# 返回
- `train_input`: (nsteps, 1, N_train)
- `train_output`: (1, 1, N_train)
- `test_input`: (nsteps, 1, N_test)
- `test_output`: (1, 1, N_test)
- `test_target_indices`: 测试预测目标对应的原始数据索引
"""
function build_windows_all_data(data_vec, train_range, test_range, nsteps, horizon=0, past_interval=0;
                                target_vec=nothing)
    # past_interval=0: 连续取值（步长=1）；否则按 past_interval 间隔采样
    input_span = past_interval == 0 ? nsteps : nsteps * past_interval
    tgt = target_vec === nothing ? data_vec : target_vec

    # 训练样本：目标 t+horizon 必须在 train_range 内
    train_input_list = []
    train_output_list = []
    train_target_indices = []
    for t in max(train_range[1], input_span + 1):(train_range[2] - horizon)
        if past_interval == 0
            push!(train_input_list, data_vec[t-nsteps:t-1])
        else
            push!(train_input_list, [data_vec[t - k * past_interval] for k in nsteps:-1:1])
        end
        push!(train_output_list, tgt[t + horizon])
        push!(train_target_indices, t + horizon)
    end

    # 测试样本：目标 t+horizon 在 test_range 内，输入可以引用训练区末尾的数据
    test_input_list = []
    test_output_list = []
    test_target_indices = []
    for t in max(test_range[1], input_span + 1):(test_range[2] - horizon)
        if past_interval == 0
            push!(test_input_list, data_vec[t-nsteps:t-1])
        else
            push!(test_input_list, [data_vec[t - k * past_interval] for k in nsteps:-1:1])
        end
        push!(test_output_list, tgt[t + horizon])
        push!(test_target_indices, t + horizon)
    end

    N_train = length(train_output_list)
    N_test = length(test_output_list)

    @assert N_train > 0 "训练样本数为0"
    @assert N_test > 0 "测试样本数为0"

    # 组装为 3D 数组 — 与峰值版本完全相同的形状
    train_input = zeros(nsteps, 1, N_train)
    train_output = zeros(1, 1, N_train)
    for i in 1:N_train
        train_input[:, 1, i] = train_input_list[i]
        train_output[1, 1, i] = train_output_list[i]
    end

    test_input = zeros(nsteps, 1, N_test)
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

"""
生成量子电路的哈密顿量和演化算符
"""
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
执行单个参数组合的训练和预测
- 数据已在 [0,π] 空间（main_optimized 中完成归一化）
- 预测在 [0,π] 空间，RMSE 反变换到 z-score 空间计算
- 返回: (output_train, output_test, variance, Ws, EEs, test_target_indices, avg_variance, train_target_indices)
"""
function train_and_predict(data_vec, train_window, test_window,
                           nmemory, nsteps, B, P, Hs, tau, batch_size,
                           data_min, data_max, horizon=0, past_interval=0)

    t0 = time()

    # 构建训练和测试窗口（数据已在 [0,π] 空间）
    train_input, train_output, test_input, test_output, test_target_indices, train_target_indices =
        build_windows_all_data(data_vec, train_window, test_window, nsteps, horizon, past_interval)
    println("  [debug] $(round(time()-t0,sigdigits=2))s | windows built: train=$(size(train_input,3)), test=$(size(test_input,3))")
    flush(stdout)

    nsystem = nsteps
    LB = length(B)
    Ws = zeros(4, LB, 1, batch_size)
    EEs = zeros(4, 1, batch_size)

    # 分布式训练+预测（合并为一轮 @spawn）
    println("  [debug] $(round(time()-t0,sigdigits=2))s | spawning $(batch_size) tasks for nqubit=$(nsteps+nmemory)...")
    flush(stdout)
    t1 = time()
    F = []
    for fp in 1:batch_size
        push!(F, Distributed.@spawn Distributed_train_and_predict(
            nmemory, nsystem, B, P, Hs[fp], tau,
            train_input, train_output, test_input))
    end
    println("  [debug] $(round(time()-t0,sigdigits=2))s | all $(batch_size) tasks spawned ($(round(time()-t1,sigdigits=2))s), waiting for results...")
    flush(stdout)

    batch_rmses = zeros(batch_size)
    best_index = 1
    best_ee = Inf
    best_train_pred = zeros(1, 1)
    best_test_pred = zeros(1, 1)
    for fp in 1:batch_size
        fp % 10 == 0 && (println("  [debug] $(round(time()-t0,sigdigits=2))s | fetching task $(fp)/$(batch_size)..."); flush(stdout))
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
    println("  [debug] $(round(time()-t0,sigdigits=2))s | all tasks fetched, avg_variance=$(round(avg_variance,digits=6))")
    flush(stdout)

    # 最优模型的预测已在 fetch 中获取
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
# 热力图生成函数（优化版本，移出循环）
# =============================================================================

function main_optimized(id; nsteps_range=1:3, nmemory_range=1:3, horizon=0, past_interval=0,
                        ham_type="linear", tau_mode="tau_nqubit", nqubit_max=11, nqubit_min=0)
    variable_name = get_variable_name(id)

    println("Processing variable: $variable_name (id=$id) [All Data Points, horizon=$horizon, past_interval=$past_interval]")

    # 读取数据
    raw_Data = readdlm(DATA_FILE, ',')

    # 数据标准化（参数仅来自训练集，避免数据泄露）
    train_raw = raw_Data[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    Data = (raw_Data .- mu) ./ sigma

    # 提取数据 min/max（从 z-score 空间的训练集计算），归一化到 [0,π]
    data_min = minimum(Data[id, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]])
    data_max = maximum(Data[id, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]])
    Data[id, :] = (Data[id, :] .- data_min) ./ (data_max - data_min) .* π

    # 确保输出目录存在（ham类型+tau+变量名+past_interval后缀区分）
    tau_label = tau_mode == "tau_1" ? "tau1" : "taun"
    config_label = "$(ham_type)_$(tau_label)"
    var_subdir = variable_name * "_$(past_interval)"
    var_dir = joinpath(OUTPUT_DIR, config_label, var_subdir)
    isdir(var_dir) || mkpath(var_dir)

    # 加载已有的 results 字典（如果有），否则初始化为空字典
    checkpoint_file = joinpath(var_dir, "results.jld2")
    if isfile(checkpoint_file)
        results = load(checkpoint_file, "results")
        println("Loaded existing results from $(checkpoint_file)")
    else
        results = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
        println("Initialized empty results dict")
    end

    # 按 nqubit 分组遍历参数组合
    t_total = time()
    nqubit_groups = Dict{Int, Vector{Tuple{Int,Int}}}()
    for nsteps in nsteps_range
        for nmemory in nmemory_range
            nq = nsteps + nmemory
            nq > nqubit_max && continue
            nqubit_min > 0 && nq < nqubit_min && continue
            if !haskey(nqubit_groups, nq)
                nqubit_groups[nq] = []
            end
            push!(nqubit_groups[nq], (nsteps, nmemory))
        end
    end

    for nq in sort(collect(keys(nqubit_groups)))
        pairs = nqubit_groups[nq]
        println("[$( Dates.format(now(), "HH:MM:SS"))] === nqubit=$nq ($(length(pairs)) combinations, NTHREADS=$NTHREADS) ===")
        flush(stdout)

        for (nsteps, nmemory) in pairs
            # Skip if already computed
            pred_file = joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory)_Past$(past_interval).jld2")
            if haskey(results, (nsteps, nmemory)) && isfile(pred_file)
                println("[$( Dates.format(now(), "HH:MM:SS"))] SKIP: nsteps=$nsteps, nmemory=$nmemory (nqubit=$nq) — already done")
                flush(stdout)
                continue
            end

            println("[$( Dates.format(now(), "HH:MM:SS"))] Processing: nsteps=$nsteps, nmemory=$nmemory (nqubit=$nq)")
            flush(stdout)

            # 量子系统配置
            tau = tau_mode == "tau_1" ? 1.0 : Float64(nq)
            B = vcat([QubitsTerm(i => "X") for i in 1:nq], [QubitsTerm(i => "Y") for i in 1:nq], [QubitsTerm(i => "Z") for i in 1:nq])
            P = BEARING_TERM_WEIGHT

            # 生成量子电路
            Hs = generate_quantum_circuits(nq, BATCH_SIZE, tau; ham_type=ham_type, seed=42)

            # 训练和预测
            local output_train, output_test, variance, _, _, test_target_indices, avg_variance, train_target_indices

            try
                output_train, output_test, variance, _, _, test_target_indices, avg_variance, train_target_indices = train_and_predict(
                    Data[id, :], TRAIN_WINDOW, TEST_WINDOW,
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
            println("  [timing] nsteps=$nsteps, nmemory=$nmemory done in $(round(time()-t_total,sigdigits=2))s")
            flush(stdout)

            # 保存预测结果（每个参数组合一个文件）
            prediction_file = joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory)_Past$(past_interval).jld2")
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
# 运行入口（优化版本）
# =============================================================================

const PAST_INTERVALS = Dict(1 => 21, 2 => 7, 3 => 34, 4 => 68)

function run_all_optimized(id; ham_type="linear", tau_mode="tau_nqubit", nqubit_max=11, nqubit_min=0,
                           nsteps_range=nsteps_val, nmemory_range=nmemory_val)
    horizon_val = 0
    past_val = past_interval_override > 0 ? past_interval_override : PAST_INTERVALS[id]
    println("Running optimized version for id=$id [ham=$ham_type, tau=$tau_mode, nqubit_max=$nqubit_max, nqubit_min=$nqubit_min, horizon=$horizon_val, past_interval=$past_val]")
    main_optimized(id, nsteps_range=nsteps_range, nmemory_range=nmemory_range, horizon=horizon_val, past_interval=past_val,
                   ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min)
end

#运行示例
if var_id > 0
    run_all_optimized(var_id, ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min)
else
    for id in 1:4
        run_all_optimized(id, ham_type=ham_type, tau_mode=tau_mode, nqubit_max=nqubit_max, nqubit_min=nqubit_min)
    end
end
