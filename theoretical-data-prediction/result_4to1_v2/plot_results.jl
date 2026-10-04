using Plots
using Plots.PlotMeasures
using JLD2
using DelimitedFiles
using Statistics

default(fontfamily="DejaVuSans", guidefontsize=13, tickfontsize=13, legendfontsize=9)

# =============================================================================
# Constants
# =============================================================================
const DATA_FILE = joinpath("..", "data", "data_theory.csv")
const TRAIN_WINDOW = (1563, 65625)
const TEST_WINDOW = (65626, 93750)
const VARIABLE_MAP = Dict("uC1" => 1, "uC2" => 2, "iL1" => 3, "iL2" => 4)
const PAST_INTERVALS = Dict("uC1" => 21, "uC2" => 7, "iL1" => 34, "iL2" => 68)

# Zoom range: 210-230μs
const ZOOM_START = 71251   # 210μs
const ZOOM_END = 82501     # 230μs

const HAM_CONFIGS = ["fc_tau1", "fc_taun", "linear_taun"]
const HAM_LABELS = Dict("fc_tau1" => "FC τ=1", "fc_taun" => "FC τ=n", "linear_taun" => "NN τ=n")
const VARIABLES = ["uC1", "uC2", "iL1", "iL2"]

# =============================================================================
# Helper functions
# =============================================================================

function load_normalized_data()
    raw = readdlm(DATA_FILE, ',')
    train_raw = raw[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    return (raw .- mu) ./ sigma
end

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
# Heatmap generation
# =============================================================================

function generate_heatmaps(results, var_name, ham, output_dir)
    nsteps_all = sort(unique(k[1] for k in keys(results)))
    nmemory_all = sort(unique(k[2] for k in keys(results)))
    ns0 = minimum(nsteps_all) - 1
    nm0 = minimum(nmemory_all) - 1
    ns_range = length(nsteps_all)
    nm_range = length(nmemory_all)

    best_mat = fill(NaN, ns_range, nm_range)
    avg_mat = fill(NaN, ns_range, nm_range)

    for ((s, m), (best_v, avg_v)) in results
        best_mat[s - ns0, m - nm0] = best_v
        avg_mat[s - ns0, m - nm0] = avg_v
    end

    title_prefix = "$(HAM_LABELS[ham]) $var_name"

    min_val, min_y, min_x = find_min(best_mat)
    p1 = heatmap(
        best_mat,
        xlabel="Memory", ylabel="Steps",
        color=:viridis, colorbar_title="Best RMSE",
        title="",
        xticks=(1:nm_range, string.(nmemory_all)),
        yticks=(1:ns_range, string.(nsteps_all)),
        grid=true, gridcolor=:black, gridlinewidth=1,
        framestyle=:box, annotations=make_annotations(best_mat),
        size=(800, 600), dpi=300, legend=:topright,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=5mm, bottom_margin=5mm,
    )
    scatter!(p1, [min_x], [min_y],
        markershape=:star5, markersize=15,
        markercolor=:red, markerstrokecolor=:red, markerstrokewidth=2,
        label="Min RMSE")
    savefig(p1, joinpath(output_dir, "$(var_name)_best_rmse.png"))
    println("  Best RMSE heatmap saved: $(var_name)_best_rmse.png")

    min_val_avg, min_y_avg, min_x_avg = find_min(avg_mat)
    p2 = heatmap(
        avg_mat,
        xlabel="Memory", ylabel="Steps",
        color=:viridis, colorbar_title="Avg RMSE",
        title="",
        xticks=(1:nm_range, string.(nmemory_all)),
        yticks=(1:ns_range, string.(nsteps_all)),
        grid=true, gridcolor=:black, gridlinewidth=1,
        framestyle=:box, annotations=make_annotations(avg_mat),
        size=(800, 600), dpi=300, legend=:topright,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=5mm, bottom_margin=5mm,
    )
    scatter!(p2, [min_x_avg], [min_y_avg],
        markershape=:star5, markersize=15,
        markercolor=:red, markerstrokecolor=:red, markerstrokewidth=2,
        label="Min RMSE")
    savefig(p2, joinpath(output_dir, "$(var_name)_avg_rmse.png"))
    println("  Avg RMSE heatmap saved: $(var_name)_avg_rmse.png")

    return min_val_avg, nsteps_all[min_y_avg], nmemory_all[min_x_avg]
end

# =============================================================================
# Prediction curve generation (v2: inverse transform from [0,π] to z-score)
# =============================================================================

function generate_prediction_curve(Data, var_id, var_name, pred_file, output_file, title_str, rmse_val, nsteps, nmemory)
    data = load(pred_file)
    output_test_0pi = data["output_test"]
    test_target_indices = data["test_target_indices"]
    d_min = data["data_min"]
    d_max = data["data_max"]

    # 反变换预测值从 [0,π] 到 z-score
    output_test = output_test_0pi ./ π .* (d_max - d_min) .+ d_min

    # Filter zoom range
    zoom_mask = (test_target_indices .>= ZOOM_START) .& (test_target_indices .<= ZOOM_END)
    zoom_indices = test_target_indices[zoom_mask]

    if isempty(zoom_indices)
        println("  Warning: No points in zoom range for $output_file")
        return
    end

    zoom_actual = Data[var_id, zoom_indices]
    zoom_pred = vec(output_test)[1:length(test_target_indices)][zoom_mask]
    p = plot(size=(800, 400), dpi=300)
    plot!(p, zoom_indices, zoom_actual,
        linecolor=:red, linewidth=1.5, alpha=0.9,
        label="Actual",
        title="",
        xticks=(71251:1250:82501, 210:2:230),
        xlabel="Time (μs)",
        ylabel=var_name,
        legend=:outertop,
        guidefontsize=13, tickfontsize=13, legendfontsize=9, tickfontcolor=:black,
        left_margin=8mm, bottom_margin=6mm,
    )
    plot!(p, zoom_indices, zoom_pred,
        linecolor=:blue, linewidth=1.5, alpha=0.7, label="Predicted (Steps=$nsteps, Memory=$nmemory)")

    savefig(p, output_file)
    println("  Prediction saved: $output_file")
end

# =============================================================================
# Main processing
# =============================================================================

function process_all()
    Data = load_normalized_data()
    output_base = "."

    summary = []

    for ham in HAM_CONFIGS
        ham == "linear_taun" || continue
        ham_dir = joinpath(output_base, ham)
        mkpath(ham_dir)

        for var in VARIABLES
            past = PAST_INTERVALS[var]
            var_dir = joinpath(ham, "$(var)_$(past)")
            results_file = joinpath(var_dir, "results.jld2")

            if !isfile(results_file)
                println("  Skipping $ham $var: no results file")
                continue
            end

            println("\n  Processing: $ham $var")
            results = load(results_file, "results")

            rmse_val, best_nsteps, best_nmemory = generate_heatmaps(
                results, var, ham, ham_dir
            )

            pred_files = [
                joinpath(var_dir, "pred_Input$(best_nsteps)_Memory$(best_nmemory)_Past$(past).jld2"),
                joinpath(var_dir, "pred_Input$(best_nsteps)_Memory$(best_nmemory).jld2")
            ]

            pred_file = nothing
            for pf in pred_files
                if isfile(pf)
                    pred_file = pf
                    break
                end
            end

            if pred_file !== nothing
                var_id = VARIABLE_MAP[var]
                curve_file = joinpath(ham_dir, "$(var)_prediction.png")
                generate_prediction_curve(
                    Data, var_id, var, pred_file, curve_file,
                    "$(HAM_LABELS[ham]) $var",
                    rmse_val, best_nsteps, best_nmemory
                )
            end

            push!(summary, (ham, var, rmse_val, best_nsteps, best_nmemory))
        end
    end

    println("\n" * "="^80)
    println("SUMMARY TABLE - 4to1 v2 (selected by avg RMSE)")
    println("="^80)
    println("Hamiltonian\tVariable\tAvg RMSE\tnsteps\tnmemory")
    println("-"^80)
    for (ham, var, rmse, ns, nm) in summary
        println("$(HAM_LABELS[ham])\t\t$var\t\t$(round(rmse, digits=4))\t$ns\t$nm")
    end
    println("="^80)
end

# Run
process_all()
