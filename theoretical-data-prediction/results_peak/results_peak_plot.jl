using Plots
using Plots.PlotMeasures
using JLD2
using CSV, DataFrames
using DelimitedFiles
using Statistics

default(fontfamily="DejaVuSans", guidefontsize=13, tickfontsize=13, legendfontsize=9)

# =============================================================================
# ODE Results Plotting - 画图脚本 (Self-contained for peak results)
# =============================================================================
# 用法：
#   include("peak_results_plot.jl")
#   plot_peak_results("results_peak/localmin/iL2", "iL2", "Local Min")
#   plot_peak_detailed("results_peak/localmin/iL2", "iL2", "Local Min", 1, 1)
# =============================================================================

# --- Constants for self-contained plotting ---
const _DATA_FILE = joinpath("data", "data_theory.csv")
const _TRAIN_WINDOW = (1563, 65625)
const _TEST_WINDOW = (65626, 93750)
const _PEAK_FILES = Dict("Local Min" => joinpath("data", "peaks_all_min.csv"), "Local Max" => joinpath("data", "peaks_all_max.csv"))
const _VARIABLE_ID_MAP = Dict("uC1" => 1, "uC2" => 2, "iL1" => 3, "iL2" => 4)

# Zoom range: 210-230μs (index range for 3.2ns interval)
const ZOOM_START = 65625   # 210μs / 3.2ns
const ZOOM_END = 71875     # 230μs / 3.2ns

# =============================================================================
# Internal helpers
# =============================================================================

function _load_normalized_data()
    raw_Data = readdlm(_DATA_FILE, ',')
    train_raw = raw_Data[:, _TRAIN_WINDOW[1]:_TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    return (raw_Data .- mu) ./ sigma
end

function _load_peak_indices(peak_label, variable_name)
    peak_file = _PEAK_FILES[peak_label]
    df = CSV.read(peak_file, DataFrame)
    return df[df.variable .== variable_name, :peak_index] .+ 1
end

function _build_target_indices(peak_indices, train_range, test_range, nsteps)
    train_pos = findall(p -> train_range[1] <= p <= train_range[2], peak_indices)
    test_pos = findall(p -> test_range[1] <= p <= test_range[2], peak_indices)

    train_target_indices = Int[]
    test_target_indices = Int[]

    for pos in train_pos
        if pos > nsteps
            push!(train_target_indices, peak_indices[pos])
        end
    end
    for pos in test_pos
        if pos > nsteps
            push!(test_target_indices, peak_indices[pos])
        end
    end

    return train_target_indices, test_target_indices
end

# =============================================================================
# Migrated dead code from Time_serial_circuit_peak.jl
# =============================================================================

function plot_prediction_results(actual_data, peak_indices_test, output_test, variance,
                                  nmemory, nsteps, variable_name, plot_range)
    p = plot(size=(800, 400), dpi=300)
    plot!(p, plot_range, actual_data[plot_range],
        label="Actual",
        title="",
        linecolor=:red, linewidth=1.5, alpha=0.9,
        xticks=(0:312.5:7813, 0:1:25),
        xlabel="Time (μs)",
        ylabel=variable_name,
        legend=:outertop,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=8mm, bottom_margin=6mm,
    )
    plot!(p, peak_indices_test, output_test',
        st=:scatter, label="Predicted (Steps=$nsteps, Memory=$nmemory)",
        markercolor=:blue, markersize=4, alpha=0.7)
    return p
end

# =============================================================================
# Heatmap helpers
# =============================================================================

function make_annotations(mat)
    ann = []
    for i in 1:size(mat, 1), j in 1:size(mat, 2)
        if !isnan(mat[i, j])
            push!(ann, (j, i, text(string(round(mat[i, j], digits=3)), 10, :center, :white)))
        end
    end
    return ann
end

function find_min(mat)
    min_val = Inf
    min_y, min_x = 0, 0
    for i in 1:size(mat, 1), j in 1:size(mat, 2)
        if !isnan(mat[i, j]) && mat[i, j] < min_val
            min_val = mat[i, j]
            min_y = i
            min_x = j
        end
    end
    return min_val, min_y, min_x
end

# =============================================================================
# Generate heatmaps (Avg RMSE only)
# =============================================================================

function generate_peak_heatmaps(results, output_dir, variable_name, peak_label)
    nsteps_all = sort(unique(k[1] for k in keys(results)))
    nmemory_all = sort(unique(k[2] for k in keys(results)))
    ns0 = minimum(nsteps_all) - 1
    nm0 = minimum(nmemory_all) - 1
    ns_range = length(nsteps_all)
    nm_range = length(nmemory_all)

    Avg_plot = fill(NaN, ns_range, nm_range)
    for ((s, m), avg_v) in results
        Avg_plot[s - ns0, m - nm0] = avg_v
    end

    avg_min_val, avg_min_y, avg_min_x = find_min(Avg_plot)
    p = heatmap(
        Avg_plot,
        xlabel="Memory", ylabel="Steps",
        color=:viridis, colorbar_title="Avg RMSE",
        title="",
        xticks=(1:nm_range, string.(nmemory_all)),
        yticks=(1:ns_range, string.(nsteps_all)),
        grid=true, gridcolor=:black, gridlinewidth=1,
        framestyle=:box, annotations=make_annotations(Avg_plot),
        size=(800, 600), dpi=300, legend=:topright,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=5mm, bottom_margin=5mm,
    )
    scatter!(p, [avg_min_x], [avg_min_y],
        markershape=:star5, markersize=15,
        markercolor=:red, markerstrokecolor=:red, markerstrokewidth=2,
        label="Min RMSE")
    savefig(p, joinpath(output_dir, "avg_rmse.png"))

    println("  Heatmap saved to $(output_dir)/")
end

# =============================================================================
# Generate detailed prediction plot (self-contained, reads CSV + jld2)
# =============================================================================

function generate_peak_detailed(output_dir, variable_name, peak_label, nsteps, nmemory)
    # Load data
    Data = _load_normalized_data()
    id = _VARIABLE_ID_MAP[variable_name]

    # Load predictions
    pred_file = joinpath(output_dir, "pred_Input$(nsteps)_Memory$(nmemory).jld2")
    if !isfile(pred_file)
        error("Prediction file not found: $(pred_file)")
    end
    data = load(pred_file)
    output_test = data["output_test"]

    # Use saved target indices from training (avoids recomputation mismatch)
    test_target_indices = data["test_target_indices"]

    # Try to load RMSE values from prediction file
    rmse_label = ""
    if haskey(data, "best_rmse") && haskey(data, "avg_rmse")
        best_rmse = data["best_rmse"]
        avg_rmse = data["avg_rmse"]
        best_index = get(data, "best_index", 0)
        rmse_label = "Best RMSE=$(round(best_rmse, digits=6)), Avg RMSE=$(round(avg_rmse, digits=6)), Ham #$(best_index)"
    end

    # Filter zoom range
    zoom_range = ZOOM_START:ZOOM_END
    zoom_mask = (test_target_indices .>= ZOOM_START) .& (test_target_indices .<= ZOOM_END)
    zoom_indices = test_target_indices[zoom_mask]

    if isempty(zoom_indices)
        error("No prediction points in zoom range $(ZOOM_START)-$(ZOOM_END) for $(variable_name)")
    end

    zoom_actual = Data[id, zoom_indices]
    zoom_pred = vec(output_test)[1:length(test_target_indices)][zoom_mask]
    errors = zoom_actual .- zoom_pred

    p = plot(size=(800, 400), dpi=300)
    plot!(p, zoom_range, Data[id, zoom_range],
        label="Actual",
        title="",
        linecolor=:red, linewidth=1.5, alpha=0.9,
        xticks=(65625:625:71875, 210:2:230),
        xlabel="Time (μs)",
        ylabel=variable_name,
        legend=:outertop,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=8mm, bottom_margin=6mm,
    )
    plot!(p, zoom_indices, zoom_actual,
        st=:scatter, label=peak_label,
        markercolor=:red, markersize=5)
    plot!(p, zoom_indices, zoom_pred,
        linecolor=:blue, linewidth=1, linestyle=:dash, alpha=0.7,
        label="")
    plot!(p, zoom_indices, zoom_pred,
        st=:scatter, label="Predicted (Steps=$nsteps, Memory=$nmemory)",
        markercolor=:blue, markersize=4, alpha=0.7)

    out_file = joinpath(output_dir, "peak_$(variable_name)_Memory$(nmemory)_Input$(nsteps).png")
    savefig(p, out_file)
    println("  Detailed plot saved to $(out_file)")
end

# =============================================================================
# Convenience entries
# =============================================================================

function plot_peak_results(var_dir, variable_name, peak_label; nsteps_range=1:6, nmemory_range=1:6)
    if !isdir(var_dir)
        error("Result directory not found: $(var_dir)")
    end

    checkpoint_file = joinpath(var_dir, "results.jld2")
    if !isfile(checkpoint_file)
        error("Checkpoint file not found: $(checkpoint_file)")
    end

    results = load(checkpoint_file, "results")
    println("Loaded results for $(variable_name) ($(peak_label)) from $(checkpoint_file)")

    # Pad missing combinations with NaN
    for s in nsteps_range, m in nmemory_range
        if !((s, m) in keys(results))
            results[(s, m)] = NaN
        end
    end

    generate_peak_heatmaps(results, var_dir, variable_name, peak_label)
end

function plot_peak_detailed(var_dir, variable_name, peak_label, nsteps, nmemory)
    if !isdir(var_dir)
        error("Result directory not found: $(var_dir)")
    end
    generate_peak_detailed(var_dir, variable_name, peak_label, nsteps, nmemory)
end
