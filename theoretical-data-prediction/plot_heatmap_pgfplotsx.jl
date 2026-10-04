######################## plot_heatmap_pgfplotsx.jl ########################
# Paper-style heatmap using PGFPlotsX (LaTeX quality output)
# Matches the style from plot_heatmap_regular_paper.jl
########################################################################

using Pkg
Pkg.activate(".")

using JLD2
using Statistics
using Printf
using Plots
using LaTeXStrings
using Plots: mm
import PGFPlotsX

pgfplotsx()

const TIMES_PREAMBLE = raw"\usepackage{newtxtext,newtxmath}"

if !(TIMES_PREAMBLE in PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, TIMES_PREAMBLE)
end

# -------------------- STYLE CONSTANTS --------------------
const GUIDE_FONT_SIZE      = 55
const TICK_FONT_SIZE       = 45
const COLORBAR_FONT_SIZE   = 45
const CELL_FONT_SIZE       = 55
const BEST_CELL_FONT_SIZE  = 55

const N_COLORBAR_TICKS = 7

# -------------------- USER SETTINGS --------------------
const FILTER_CONFIG = "4to1"   # "1to1", "3to1", "4to1"
const FILTER_HAM = "linear_taun"

const VARIABLES = if FILTER_CONFIG == "3to1"
    ["uC1", "iL1", "iL2"]
else
    ["uC1", "uC2", "iL1", "iL2"]
end

const PAST_INTERVALS = Dict("uC1" => 21, "uC2" => 7, "iL1" => 34, "iL2" => 68)

const OUTPUT_DIR = joinpath("paper_figures", "fig2_pgfplotsx")
mkpath(OUTPUT_DIR)

# -------------------- HELPERS --------------------

same_float(a::Real, b::Real; atol::Real = 1e-9) = abs(Float64(a) - Float64(b)) <= atol

function tile_value(x::Real)
    if !isfinite(x)
        return L"\mathrm{NaN}"
    end
    return latexstring(@sprintf("%.2f", Float64(x)))
end

function colorbar_value(x::Real)
    if !isfinite(x)
        return L"\mathrm{NaN}"
    end
    return latexstring(@sprintf("%.2f", Float64(x)))
end

function rect_edges(center::Int)
    return center - 0.5, center + 0.5
end

# -------------------- HEATMAP GENERATION --------------------

function generate_pgfplotsx_heatmap(
    results::Dict,
    var_name::String,
    output_path::String;
    use_avg::Bool = true
)
    # Extract axes (x = nmemory, y = nsteps)
    xvals = sort(unique(k[2] for k in keys(results)))  # nmemory
    yvals = sort(unique(k[1] for k in keys(results)))  # nsteps
    
    # Build matrix
    Z = fill(NaN, length(yvals), length(xvals))
    for ((s, m), (best_v, avg_v)) in results
        xi = findfirst(==(m), xvals)  # m = nmemory
        yi = findfirst(==(s), yvals)  # s = nsteps
        if xi !== nothing && yi !== nothing
            Z[yi, xi] = use_avg ? avg_v : best_v
        end
    end
    
    # Find best cell
    finite_pos = findall(isfinite, Z)
    isempty(finite_pos) && error("All values are NaN for $var_name")
    
    best_cart = finite_pos[argmin(Z[finite_pos])]
    best_i, best_j = Tuple(best_cart)
    best_x = xvals[best_j]
    best_y = yvals[best_i]
    best_rmse = Z[best_i, best_j]
    
    # Colorbar range
    finite_z = Z[isfinite.(Z)]
    zmin, zmax = extrema(finite_z)
    
    # Labels
    x_label = L"N_{\mathrm{mem}}"
    y_label = L"N_{\mathrm{steps}}"
    metric_label = "RMSE"
    
    # Colorbar ticks
    colorbar_tick_values =
        same_float(zmin, zmax) ?
        [zmin] :
        collect(range(zmin, zmax; length=N_COLORBAR_TICKS))
    
    colorbar_yticks = join(
        [@sprintf("%.12g", value) for value in colorbar_tick_values],
        ","
    )
    
    colorbar_yticklabels = join(
        [@sprintf("%.2f", value) for value in colorbar_tick_values],
        ","
    )
    
    colorbar_style =
        "{ytick={" * colorbar_yticks * "}," *
        "yticklabels={" * colorbar_yticklabels * "}," *
        "yticklabel style={font=\\fontsize{" *
        string(TICK_FONT_SIZE) * "}{" *
        string(TICK_FONT_SIZE + 7) *
        "}\\selectfont}," *
        "title={" * metric_label * "}," *
        "title style={font=\\fontsize{" *
        string(COLORBAR_FONT_SIZE) * "}{" *
        string(COLORBAR_FONT_SIZE + 8) *
        "}\\selectfont, at={(0.5,1.03)}, " *
        "anchor=south, rotate=0}}"
    
    x_tick_labels = latexstring.(string.(xvals))
    y_tick_labels = latexstring.(string.(yvals))
    
    # Create heatmap
    p = heatmap(
        xvals,
        yvals,
        Z;
        xlabel = x_label,
        ylabel = y_label,
        title = "",
        colorbar_title = "",
        clims = (zmin, zmax),
        xticks = (xvals, x_tick_labels),
        yticks = (yvals, y_tick_labels),
        framestyle = :axes,
        grid = false,
        size = (1000, 800),
        left_margin = 9mm,
        right_margin = 20mm,
        bottom_margin = 1mm,
        top_margin = 0mm,
        guidefont = font(GUIDE_FONT_SIZE),
        tickfont = font(TICK_FONT_SIZE),
        extra_kwargs = Dict(
            :subplot => Dict(
                "colorbar style" => colorbar_style,
            ),
        ),
    )
    
    # Black border
    x_left   = minimum(xvals) - 0.5
    x_right  = maximum(xvals) + 0.5
    y_bottom = minimum(yvals) - 0.5
    y_top    = maximum(yvals) + 0.5
    
    plot!(
        p,
        [x_left, x_right, x_right, x_left, x_left],
        [y_bottom, y_bottom, y_top, y_top, y_bottom];
        color = :black,
        linewidth = 3,
        label = false,
    )
    
    # Cell values (white text, rotated)
    for (i, yv) in enumerate(yvals)
        for (j, xv) in enumerate(xvals)
            if isfinite(Z[i, j])
                txt_color = :white
                txt_size = CELL_FONT_SIZE
                if i == best_i && j == best_j
                    txt_color = :yellow
                    txt_size = BEST_CELL_FONT_SIZE
                end
                annotate!(
                    p,
                    xv,
                    yv,
                    text(
                        tile_value(Z[i, j]),
                        txt_size,
                        txt_color,
                        :center;
                        rotation = 90,
                    )
                )
            end
        end
    end
    
    # Highlight best cell with yellow border
    x0, x1 = rect_edges(best_x)
    y0, y1 = rect_edges(best_y)
    
    plot!(p, [x0, x1], [y0, y0], color=:yellow, lw=5, label=false)
    plot!(p, [x1, x1], [y0, y1], color=:yellow, lw=5, label=false)
    plot!(p, [x1, x0], [y1, y1], color=:yellow, lw=5, label=false)
    plot!(p, [x0, x0], [y1, y0], color=:yellow, lw=5, label=false)
    
    savefig(p, output_path)
    println("  Saved: $output_path")
    
    return best_rmse, best_x, best_y
end

# -------------------- MAIN --------------------

function main()
    println("=" ^ 60)
    println("PGFPlotsX Paper-style Heatmap Generation")
    println("Config: $FILTER_CONFIG, Ham: $FILTER_HAM")
    println("=" ^ 60)
    
    for var in VARIABLES
        past = PAST_INTERVALS[var]
        var_dir = joinpath("result_$(FILTER_CONFIG)_v2", FILTER_HAM, "$(var)_$(past)")
        results_file = joinpath(var_dir, "results.jld2")
        
        if !isfile(results_file)
            println("  Skipping $var: no results file")
            continue
        end
        
        println("\nProcessing: $var")
        results = load(results_file, "results")
        
        # Generate avg RMSE heatmap only
        files_str = join(VARIABLES, "-")
        output_path = joinpath(OUTPUT_DIR, "heatmap_rmse_mem_vs_nt_source_theory_files_$(files_str)_target_$(var)_config_$(FILTER_CONFIG)_ham_$(FILTER_HAM).png")
        try
            best_rmse, best_x, best_y = generate_pgfplotsx_heatmap(
                results, var, output_path; use_avg = true
            )
            println("  Best: Steps=$best_y, Memory=$best_x, RMSE=$(round(best_rmse, digits=4))")
        catch e
            println("  Error: $e")
        end
    end
    
    println("\n" * "=" ^ 60)
    println("Done! Output: $OUTPUT_DIR")
    println("=" ^ 60)
end

main()
