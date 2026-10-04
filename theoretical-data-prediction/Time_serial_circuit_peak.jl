push!(LOAD_PATH,"../QuantumCircuits/src","../VQC/src","../QuantumReservoirComputing/src")
using Pkg
Pkg.activate(".")
using Random
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

include("results_peak/results_peak_plot.jl")

# =============================================================================
# 配置参数
# =============================================================================

# 分布式计算配置
#const PID = min(450, floor(Int, Sys.CPU_THREADS * 0.2))
const PID = 100
const BLAS_THREADS = 6

# 数据配置
const DATA_FILE = joinpath("data", "data_theory.csv")

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
const PLOT_RANGE = 62500:93750       # 绘图范围

# 量子系统参数
const BATCH_SIZE = 100          # 批次大小
const BEARING_TERM_WEIGHT = 0   # 噪声参数

# =============================================================================
# 环境初始化
# =============================================================================

# 启动分布式进程
nprocs() < PID && addprocs(PID - nprocs() + 1)
println("The number of processors: ", nprocs())

# 配置分布式环境
@everywhere begin
    include("src/head.jl")
    using LinearAlgebra
    BLAS.set_num_threads(1)
end
BLAS.set_num_threads(BLAS_THREADS)

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
# PeakType 枚举与配置
# =============================================================================

@enum PeakType MIN_PEAKS MAX_PEAKS

const PEAK_FILES = Dict(MIN_PEAKS => joinpath("data", "peaks_all_min.csv"), MAX_PEAKS => joinpath("data", "peaks_all_max.csv"))
const OUTPUT_DIRS = Dict(MIN_PEAKS => joinpath("results_peak", "localmin"), MAX_PEAKS => joinpath("results_peak", "localmax"))
const PEAK_LABELS = Dict(MIN_PEAKS => "Local Min", MAX_PEAKS => "Local Max")

# =============================================================================
# 数据处理工具函数
# =============================================================================

"""
从CSV中提取指定变量的峰值索引
"""
function extract_peak_indices(df, variable)
    return df[df.variable .== variable, :peak_index] .+ 1
end

"""
基于峰值索引范围构建训练/测试滑动窗口

# 参数
- `peak_data`: 峰值数据向量（已标准化）
- `peak_indices`: 峰值对应的原始数据索引（按时间排序）
- `train_range`: 训练时间范围 (start, end)，按原始数据索引
- `test_range`: 测试时间范围 (start, end)，按原始数据索引
- `nsteps`: 输入步数（历史峰值个数）

# 返回
- `train_input`: (nsteps, 1, N_train)
- `train_output`: (1, 1, N_train)
- `test_input`: (nsteps, 1, N_test)
- `test_output`: (1, 1, N_test)
- `test_target_indices`: 测试预测目标对应的原始数据索引
"""
function build_windows_by_peak_range(peak_data, peak_indices, train_range, test_range, nsteps)
    train_pos = findall(p -> train_range[1] <= p <= train_range[2], peak_indices)
    test_pos = findall(p -> test_range[1] <= p <= test_range[2], peak_indices)

    @assert !isempty(train_pos) "训练窗口内没有峰值点"
    @assert !isempty(test_pos) "测试窗口内没有峰值点"

    # 训练样本
    train_input_list = []
    train_output_list = []
    train_target_indices = []
    for pos in train_pos
        if pos > nsteps
            push!(train_input_list, peak_data[pos-nsteps:pos-1])
            push!(train_output_list, peak_data[pos])
            push!(train_target_indices, peak_indices[pos])
        end
    end

    # 测试样本
    test_input_list = []
    test_output_list = []
    test_target_indices = []
    for pos in test_pos
        if pos > nsteps
            push!(test_input_list, peak_data[pos-nsteps:pos-1])
            push!(test_output_list, peak_data[pos])
            push!(test_target_indices, peak_indices[pos])
        end
    end

    N_train = length(train_output_list)
    N_test = length(test_output_list)

    @assert N_train > 0 "训练样本数为0"
    @assert N_test > 0 "测试样本数为0"

    # 组装为 3D 数组
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
- 返回: (训练集预测结果, 测试集预测结果, 方差, 权重, 误差)
"""
function train_and_predict(peak_data, peak_indices, train_window, test_window,
                           nmemory, nsteps, B, P, Hs, tau, batch_size,
                           original_data, peak_data_min, peak_data_max)

    # 构建训练和测试窗口
    train_input, train_output, test_input, test_output, test_target_indices, train_target_indices =
        build_windows_by_peak_range(peak_data, peak_indices, train_window, test_window, nsteps)

    # 将 train_input 和 test_input 归一化到 [0, π]（用于量子电路编码）
    train_input = (train_input .- peak_data_min) ./ (peak_data_max - peak_data_min) .* π
    test_input = (test_input .- peak_data_min) ./ (peak_data_max - peak_data_min) .* π

    
    nsystem = nsteps
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
        n_compare = min(length(original_data[test_target_indices]), size(test_pred, 2))
        batch_rmses[fp] = sqrt(mean((original_data[test_target_indices][1:n_compare] .- vec(test_pred)[1:n_compare]).^2))
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

    # 计算best模型的RMSE（标准化后）
    n_test = size(output_test, 2)
    test_actual = original_data[test_target_indices]
    pred_test = vec(output_test)
    n_compare = min(length(test_actual), n_test)
    variance = sqrt(mean((test_actual[1:n_compare] .- pred_test[1:n_compare]).^2))

    return output_train, output_test, variance, Ws, EEs, test_target_indices, avg_variance, train_target_indices, best_index
end

# =============================================================================
# 可视化函数
# =============================================================================

"""
绘制预测结果对比图
"""
function plot_prediction_results(actual_data, peak_indices_test, output_test, variance, 
                                  nmemory, nsteps, variable_name, plot_range)
    plot(
        plot_range, actual_data[plot_range],
        label="Actual, RMSE=$(round(variance, digits=6))",
        title="Memory=$(nmemory), Input=$(nsteps)",
        size=(800, 400),
        guidefontsize=12,
        tickfontsize=12,
        legendfontsize=12
    )
    plot!(peak_indices_test, actual_data[peak_indices_test], st=:scatter, label="Local Min")
    plot!(
        peak_indices_test, output_test',
        markershape=:circle,
        label="Predict",
        xticks=(0:312.5:7813, 0:1:25),
        legend=:topleft,
        ylabel=variable_name,
        xlabel="μs",
        bottom_margin=4mm,
        left_margin=4mm
    )
end

# =============================================================================
# 主程序
# =============================================================================

function main(id, peak_type::PeakType; nsteps_range=1:6, nmemory_range=1:6)
    # 获取变量名和配置
    variable_name = get_variable_name(id)
    peak_file = PEAK_FILES[peak_type]
    output_dir = OUTPUT_DIRS[peak_type]
    peak_label = PEAK_LABELS[peak_type]

    println("Processing variable: $variable_name (id=$id), peak_type=$peak_type")

    # 读取数据
    df = CSV.read(peak_file, DataFrame)
    raw_Data = readdlm(DATA_FILE, ',')

    # Z-score 标准化 —— 只用训练集原始数据计算 mu/sigma，避免数据泄露
    train_raw = raw_Data[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    Data = (raw_Data .- mu) ./ sigma

    # 提取峰值索引
    peak_indices = extract_peak_indices(df, variable_name)

    # 提取峰值数据
    peak_data = Data[id, peak_indices]
    # [0,1] 归一化参数 —— 只用训练集内的 peak Z-score 数据
    train_peak_mask = TRAIN_WINDOW[1] .<= peak_indices .<= TRAIN_WINDOW[2]
    peak_data_min = minimum(peak_data[train_peak_mask])
    peak_data_max = maximum(peak_data[train_peak_mask])

    # 确保输出目录存在
    var_dir = joinpath(output_dir, variable_name)
    isdir(var_dir) || mkpath(var_dir)

    # 加载已有的 results 字典（如果有），否则初始化为空字典
    checkpoint_file = joinpath(var_dir, "results.jld2")
    if isfile(checkpoint_file)
        results = load(checkpoint_file, "results")
        println("Loaded existing results from $(checkpoint_file)")
    else
        results = Dict{Tuple{Int,Int}, Float64}()
        println("Initialized empty results dict")
    end

    # 遍历指定的参数组合
    for nsteps in nsteps_range
        for nmemory in nmemory_range
            println("Processing: nsteps=$nsteps, nmemory=$nmemory")

            # 量子系统配置
            nqubit = nsteps + nmemory
            tau = nqubit / 1  # 演化时间
            B = vcat([QubitsTerm(i => "X") for i in 1:nqubit], [QubitsTerm(i => "Y") for i in 1:nqubit], [QubitsTerm(i => "Z") for i in 1:nqubit])
            P = BEARING_TERM_WEIGHT

            # 生成量子电路 (Ham linear = nearest-neighbor XYZ)
            Hs = generate_quantum_circuits(nqubit, BATCH_SIZE, tau, ham_type="linear")

            # 训练和预测
            output_train, output_test, variance, _, _, test_target_indices, avg_variance, train_target_indices, best_index = train_and_predict(
                peak_data, peak_indices, TRAIN_WINDOW, TEST_WINDOW,
                nmemory, nsteps, B, P, Hs, tau, BATCH_SIZE,
                Data[id, :], peak_data_min, peak_data_max
            )

            results[(nsteps, nmemory)] = avg_variance
            println("  Best RMSE: $variance, Avg RMSE: $avg_variance, Best Index: $best_index")

            # 保存预测结果（每个参数组合一个文件）
            prediction_file = joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory).jld2")
            save(prediction_file,
                "output_test", output_test,
                "test_target_indices", test_target_indices,
                "output_train", output_train,
                "train_target_indices", train_target_indices,
                "best_index", best_index,
                "avg_rmse", avg_variance,
                "best_rmse", variance)

            # 保存 checkpoint
            save(checkpoint_file, "results", results)
        end
    end

    println("Done! Results saved to $(var_dir)/")
end

# =============================================================================
# 运行入口
# =============================================================================

function run_all(id)
    for pt in instances(PeakType)
        for nsteps in 1:9
            for nmemory in 1:10-nsteps
                println("Running for id=$id, peak_type=$pt, nsteps=$nsteps, nmemory=$nmemory")
                main(id, pt, nsteps_range=nsteps:nsteps, nmemory_range=nmemory:nmemory)
            end
        end
        #main(id, pt, nsteps_range=1:6, nmemory_range=1:6)
    end
end

for id in 1:4
    run_all(id)
end

# 运行单个变量的极小值预测
#main(1, MIN_PEAKS)

# 运行单个变量的极大值预测
# main(1, MAX_PEAKS)

# 运行单个变量的两种峰值预测
# run_all(1)

# 运行所有变量（默认计算全部 1:6 × 1:6）
# 可以指定子集，例如只算 nsteps=6:6, nmemory=1:6：
# main(1, MIN_PEAKS, nsteps_range=6:6, nmemory_range=1:6)
# for id in 1:4
#     run_all(id)
# end