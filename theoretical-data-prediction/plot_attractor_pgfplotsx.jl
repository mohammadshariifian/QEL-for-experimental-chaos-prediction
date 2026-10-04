######################## plot_attractor_pgfplotsx.jl ########################
# Paper-style attractor plots using PGFPlotsX
# Style matched to plot_heatmap_regular_paper.jl
########################################################################

using Pkg
Pkg.activate(".")

using JLD2
using Statistics
using Printf
using DelimitedFiles
using Plots
using LaTeXStrings
using Plots: mm
import PGFPlotsX

pgfplotsx()

const TIMES_PREAMBLE = raw"\usepackage{newtxtext,newtxmath}"

if !(TIMES_PREAMBLE in PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, TIMES_PREAMBLE)
end

# -------------------- STYLE CONSTANTS (matched to paper script) --------------------
const ATTRACTOR_GUIDE_FONT_SIZE  = 40
const ATTRACTOR_TICK_FONT_SIZE   = 35
const ATTRACTOR_LEGEND_FONT_SIZE = 28

const ATTRACTOR_LAG_STEPS = 30  # lag in index units

# Zoom range: 210-230μs
const ZOOM_START = 71251
const ZOOM_END = 82501

# -------------------- USER SETTINGS --------------------
const FILTER_CONFIG = "4to1"
const FILTER_HAM = "linear_taun"
const PAST_INTERVALS = Dict("uC1" => 21, "uC2" => 7, "iL1" => 34, "iL2" => 68)

const VARIABLES = if FILTER_CONFIG == "3to1"
    ["uC1", "iL1", "iL2"]
else
    ["uC1", "uC2", "iL1", "iL2"]
end

# Variable map for data loading
const VARIABLE_MAP = Dict("uC1" => 1, "uC2" => 2, "iL1" => 3, "iL2" => 4)

# Cross-attractor pairs
const CROSS_PAIRS = [
    ("iL1", "uC1"),
    ("iL2", "uC2"),
    ("iL2", "iL1"),
]

# Past interval (var_x, var_y) for each cross pair. The variables of a pair are
# predicted with the same interval (and best config) so the two trajectories are
# directly comparable; matches the *_34 / *_68 companion directories.
const CROSS_PAST = Dict(
    ("iL1", "uC1") => (34, 34),   # uC1_34 x iL1_34
    ("iL2", "uC2") => (68, 7),    # iL2_68 x uC2_7  (no uC2 companion)
    ("iL2", "iL1") => (68, 68),   # iL2_68 x iL1_68
)

const DATA_FILE = joinpath("data", "data_theory.csv")
const TRAIN_WINDOW = (1563, 65625)

const OUTPUT_DIR = joinpath("paper_figures", "fig4_pgfplotsx")
mkpath(OUTPUT_DIR)

# -------------------- HELPERS --------------------

function load_normalized_data()
    raw = readdlm(DATA_FILE, ',')
    train_raw = raw[:, TRAIN_WINDOW[1]:TRAIN_WINDOW[2]]
    mu = mean(train_raw, dims=2)
    sigma = std(train_raw, dims=2)
    return (raw .- mu) ./ sigma
end

function feature_latex_symbol(var_name)
    var_name == "uC1" && return raw"u_{C_1}"
    var_name == "uC2" && return raw"u_{C_2}"
    var_name == "iL1" && return raw"i_{L_1}"
    var_name == "iL2" && return raw"i_{L_2}"
    return raw"\mathrm{" * var_name * "}"
end

function attractor_axis_label(var_name; lag_steps=nothing)
    symbol = feature_latex_symbol(var_name)
    if lag_steps === nothing
        return latexstring(symbol * raw"(t)")
    end
    return latexstring(symbol * raw"(t+" * string(lag_steps) * raw"\Delta t)")
end

function attractor_base_plot(xlabel_latex, ylabel_latex)
    return plot(
        xlabel = xlabel_latex,
        ylabel = ylabel_latex,
        title = "",
        framestyle = :box,
        grid = true,
        size = (900, 800),
        left_margin = 16mm,
        right_margin = 8mm,
        bottom_margin = 14mm,
        top_margin = 5mm,
        guidefont = font(ATTRACTOR_GUIDE_FONT_SIZE),
        tickfont = font(ATTRACTOR_TICK_FONT_SIZE),
        legendfont = font(ATTRACTOR_LEGEND_FONT_SIZE),
        legend = :topright,
        extra_kwargs = Dict(
            :subplot => Dict(
                "axis line style" => "{black, line width=1.2pt}",
                "legend image post style" => "{mark=none}",
            ),
        ),
    )
end

function delay_pairs_by_lag(y, lag_steps)
    yy = Float64.(vec(y))
    
    lag_steps >= 1 || error("lag_steps must be >= 1")
    
    if lag_steps >= length(yy)
        @warn "Attractor lag larger than time series"
        return Float64[], Float64[]
    end
    
    xvals = Float64[]
    yvals = Float64[]
    for k in 1:(length(yy) - lag_steps)
        if isfinite(yy[k]) && isfinite(yy[k + lag_steps])
            push!(xvals, yy[k])
            push!(yvals, yy[k + lag_steps])
        end
    end
    
    return xvals, yvals
end

# -------------------- ATTRACTOR GENERATION --------------------

function generate_attractor(Data, var_name, pred_file, output_path)
    data = load(pred_file)
    output_test_0pi = data["output_test"]
    test_target_indices = data["test_target_indices"]
    d_min = data["data_min"]
    d_max = data["data_max"]
    
    # Inverse transform
    output_test = output_test_0pi ./ π .* (d_max - d_min) .+ d_min
    
    # Filter zoom range
    zoom_mask = (test_target_indices .>= ZOOM_START) .& (test_target_indices .<= ZOOM_END)
    zoom_indices = test_target_indices[zoom_mask]
    
    isempty(zoom_indices) && return nothing
    
    var_id = VARIABLE_MAP[var_name]
    zoom_actual = Data[var_id, zoom_indices]
    zoom_pred = vec(output_test)[1:length(test_target_indices)][zoom_mask]
    
    # Delay embedding
    x_true, y_true = delay_pairs_by_lag(zoom_actual, ATTRACTOR_LAG_STEPS)
    x_pred, y_pred = delay_pairs_by_lag(zoom_pred, ATTRACTOR_LAG_STEPS)
    
    (isempty(x_true) || isempty(x_pred)) && return nothing
    
    xlabel_latex = attractor_axis_label(var_name; lag_steps=ATTRACTOR_LAG_STEPS)
    ylabel_latex = attractor_axis_label(var_name)
    
    # Left panel: Actual
    p1 = attractor_base_plot(xlabel_latex, ylabel_latex)
    scatter!(p1, y_true, x_true;
        markersize = 1.5,
        markerstrokewidth = 0,
        markerstrokecolor = :transparent,
        markercolor = :blue,
        seriesalpha = 0.70,
        label = latexstring(raw"\mathrm{True}"),
    )
    
    # Right panel: Predicted
    p2 = attractor_base_plot(xlabel_latex, ylabel_latex)
    scatter!(p2, y_pred, x_pred;
        markersize = 1.5,
        markerstrokewidth = 0,
        markerstrokecolor = :transparent,
        markercolor = :red,
        seriesalpha = 0.70,
        label = latexstring(raw"\mathrm{Predicted}"),
    )
    
    # Combine side by side
    p = plot(p1, p2, layout=(1, 2), size=(1800, 800))
    
    savefig(p, output_path)
    println("    Saved: $(basename(output_path))")
    return output_path
end

function load_pred_values(pred_file, zoom_mask_len)
    data = load(pred_file)
    output_test_0pi = data["output_test"]
    test_target_indices = data["test_target_indices"]
    d_min = data["data_min"]
    d_max = data["data_max"]
    
    output_test = output_test_0pi ./ π .* (d_max - d_min) .+ d_min
    
    zoom_mask = (test_target_indices .>= ZOOM_START) .& (test_target_indices .<= ZOOM_END)
    zoom_indices = test_target_indices[zoom_mask]
    
    isempty(zoom_indices) && return nothing, nothing
    
    zoom_pred = vec(output_test)[1:length(test_target_indices)][zoom_mask]
    return zoom_indices, zoom_pred
end

function generate_cross_attractor(Data, var_x, var_y, pred_file_x, pred_file_y, output_path)
    x_id = VARIABLE_MAP[var_x]
    y_id = VARIABLE_MAP[var_y]
    
    # Load predictions for each variable separately
    zoom_indices_x, zoom_pred_x = load_pred_values(pred_file_x, 0)
    zoom_indices_y, zoom_pred_y = load_pred_values(pred_file_y, 0)
    
    (zoom_indices_x === nothing || zoom_indices_y === nothing) && return nothing
    
    # Use common zoom indices
    common_indices = intersect(Set(zoom_indices_x), Set(zoom_indices_y))
    isempty(common_indices) && return nothing
    common_indices = sort(collect(common_indices))
    
    # Build index maps
    idx_map_x = Dict(v => i for (i, v) in enumerate(zoom_indices_x))
    idx_map_y = Dict(v => i for (i, v) in enumerate(zoom_indices_y))
    
    zoom_actual_x = Data[x_id, common_indices]
    zoom_actual_y = Data[y_id, common_indices]
    
    pred_x = [zoom_pred_x[idx_map_x[i]] for i in common_indices]
    pred_y = [zoom_pred_y[idx_map_y[i]] for i in common_indices]
    
    xlabel_latex = attractor_axis_label(var_x)
    ylabel_latex = attractor_axis_label(var_y)
    
    # Left panel: Actual
    p1 = attractor_base_plot(xlabel_latex, ylabel_latex)
    scatter!(p1, zoom_actual_x, zoom_actual_y;
        markersize = 1.5,
        markerstrokewidth = 0,
        markercolor = :blue,
        seriesalpha = 0.70,
        label = latexstring(raw"\mathrm{True}"),
    )
    
    # Right panel: Predicted
    p2 = attractor_base_plot(xlabel_latex, ylabel_latex)
    scatter!(p2, pred_x, pred_y;
        markersize = 1.5,
        markerstrokewidth = 0,
        markercolor = :red,
        seriesalpha = 0.70,
        label = latexstring(raw"\mathrm{Predicted}"),
    )
    
    # Combine side by side
    p = plot(p1, p2, layout=(1, 2), size=(1800, 800))
    
    savefig(p, output_path)
    println("    Saved: $(basename(output_path))")
    return output_path
end

# -------------------- MAIN --------------------

function main()
    println("=" ^ 60)
    println("PGFPlotsX Paper-style Attractor Generation")
    println("Config: $FILTER_CONFIG, Ham: $FILTER_HAM")
    println("=" ^ 60)
    
    Data = load_normalized_data()
    
    for var in VARIABLES
        past = PAST_INTERVALS[var]
        var_dir = joinpath("result_$(FILTER_CONFIG)_v2", FILTER_HAM, "$(var)_$(past)")
        results_file = joinpath(var_dir, "results.jld2")
        
        if !isfile(results_file)
            println("  Skipping $var: no results file")
            continue
        end
        
        results = load(results_file, "results")
        
        # Find best (nsteps, nmemory) by avg RMSE
        best_key = nothing
        best_val = Inf
        for (k, v) in results
            if !isnan(v[2]) && v[2] < best_val
                best_val = v[2]
                best_key = k
            end
        end
        
        best_key === nothing && continue
        nsteps, nmemory = best_key
        
        println("\nProcessing: $var (Steps=$nsteps, Memory=$nmemory, Avg RMSE=$(round(best_val, digits=4)))")
        
        # Find pred file
        pred_files = [
            joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory)_Past$(past).jld2"),
            joinpath(var_dir, "pred_Input$(nsteps)_Memory$(nmemory).jld2"),
        ]
        
        pred_file = nothing
        for pf in pred_files
            if isfile(pf)
                pred_file = pf
                break
            end
        end
        
        pred_file === nothing && continue
        
        # Single attractor
        output_path = joinpath(OUTPUT_DIR, "attractor_$(FILTER_CONFIG)_$(var)_$(FILTER_HAM).png")
        try
            generate_attractor(Data, var, pred_file, output_path)
        catch e
            println("    Error: $e")
        end
    end
    
    # Cross attractors - need separate prediction files for each variable
    println("\n--- Cross Attractors ---")
    for (var_x, var_y) in CROSS_PAIRS
        (FILTER_CONFIG == "3to1" && (var_x == "uC2" || var_y == "uC2")) && continue
        
        # Past intervals: use the matched pair when defined, else the variable default
        past_x, past_y = get(CROSS_PAST, (var_x, var_y),
                             (PAST_INTERVALS[var_x], PAST_INTERVALS[var_y]))

        # Find best prediction file for var_x
        var_dir_x = joinpath("result_$(FILTER_CONFIG)_v2", FILTER_HAM, "$(var_x)_$(past_x)")
        results_file_x = joinpath(var_dir_x, "results.jld2")
        
        # Find best prediction file for var_y
        var_dir_y = joinpath("result_$(FILTER_CONFIG)_v2", FILTER_HAM, "$(var_y)_$(past_y)")
        results_file_y = joinpath(var_dir_y, "results.jld2")
        
        if !isfile(results_file_x) || !isfile(results_file_y)
            println("  Skipping $var_x-$var_y: no results file")
            continue
        end
        
        # Find best combo for var_x
        results_x = load(results_file_x, "results")
        best_key_x = nothing
        best_val_x = Inf
        for (k, v) in results_x
            if !isnan(v[2]) && v[2] < best_val_x
                best_val_x = v[2]
                best_key_x = k
            end
        end
        
        # Find best combo for var_y
        results_y = load(results_file_y, "results")
        best_key_y = nothing
        best_val_y = Inf
        for (k, v) in results_y
            if !isnan(v[2]) && v[2] < best_val_y
                best_val_y = v[2]
                best_key_y = k
            end
        end
        
        (best_key_x === nothing || best_key_y === nothing) && continue
        nsteps_x, nmemory_x = best_key_x
        nsteps_y, nmemory_y = best_key_y
        
        # Find pred file for var_x
        pred_files_x = [
            joinpath(var_dir_x, "pred_Input$(nsteps_x)_Memory$(nmemory_x)_Past$(past_x).jld2"),
            joinpath(var_dir_x, "pred_Input$(nsteps_x)_Memory$(nmemory_x).jld2"),
        ]
        pred_file_x = nothing
        for pf in pred_files_x
            if isfile(pf)
                pred_file_x = pf
                break
            end
        end
        
        # Find pred file for var_y
        pred_files_y = [
            joinpath(var_dir_y, "pred_Input$(nsteps_y)_Memory$(nmemory_y)_Past$(past_y).jld2"),
            joinpath(var_dir_y, "pred_Input$(nsteps_y)_Memory$(nmemory_y).jld2"),
        ]
        pred_file_y = nothing
        for pf in pred_files_y
            if isfile(pf)
                pred_file_y = pf
                break
            end
        end
        
        (pred_file_x === nothing || pred_file_y === nothing) && continue
        
        println("  $var_x-$var_y ($var_x: S=$nsteps_x,M=$nmemory_x; $var_y: S=$nsteps_y,M=$nmemory_y)")
        output_path = joinpath(OUTPUT_DIR, "cross_attractor_$(FILTER_CONFIG)_$(var_x)_$(var_y)_$(FILTER_HAM).png")
        try
            generate_cross_attractor(Data, var_x, var_y, pred_file_x, pred_file_y, output_path)
        catch e
            println("    Error: $e")
        end
    end
    
    println("\n" * "=" ^ 60)
    println("Done! Output: $OUTPUT_DIR")
    println("=" ^ 60)
end

main()
