######################## plot_peak_heatmap_pgfplotsx.jl ########################
# Paper-style heatmap for peak/maplike results using PGFPlotsX
# Style matched to plot_heatmap_maplike_paper.jl
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

# -------------------- STYLE CONSTANTS (matched to maplike script) --------------------
const GUIDE_FONT_SIZE      = 55
const TICK_FONT_SIZE       = 45
const COLORBAR_FONT_SIZE   = 45
const CELL_FONT_SIZE       = 25
const BEST_CELL_FONT_SIZE  = 28
const TITLE_FONT_SIZE      = 55

const N_COLORBAR_TICKS = 7

# -------------------- USER SETTINGS --------------------
const FILTER_CONFIG = "1to1"   # "1to1", "3to1", "4to1"
const PEAK_TYPES = ["localmin", "localmax"]

const VARIABLES = if FILTER_CONFIG == "3to1"
    ["uC1", "iL1", "iL2"]
else
    ["uC1", "uC2", "iL1", "iL2"]
end

const OUTPUT_DIR = joinpath("paper_figures", "fig5_pgfplotsx")
mkpath(OUTPUT_DIR)

# -------------------- HELPERS --------------------

same_float(a::Real, b::Real; atol::Real = 1e-9) = abs(Float64(a) - Float64(b)) <= atol

function tile_value(x::Real)
    if !isfinite(x)
        return L"\mathrm{NaN}"
    end
    return latexstring(@sprintf("%.2f", Float64(x)))
end

function rect_edges(center::Int)
    return center - 0.5, center + 0.5
end

function feature_title(var_name, peak_type)
    feature_tex =
        var_name == "uC1" ? raw"u_{C_1}" :
        var_name == "uC2" ? raw"u_{C_2}" :
        var_name == "iL1" ? raw"i_{L_1}" :
        var_name == "iL2" ? raw"i_{L_2}" :
                            raw"\mathrm{" * var_name * "}"

    extrema_tex =
        peak_type == "localmin" ? raw"\mathrm{minima}" :
        peak_type == "localmax" ? raw"\mathrm{maxima}" :
                                  raw"\mathrm{extrema}"

    return latexstring(feature_tex * raw"\;" * extrema_tex)
end

# -------------------- HEATMAP GENERATION --------------------

function generate_peak_heatmap(
    results::Dict{Tuple{Int,Int}, Float64},
    var_name::String,
    peak_type::String,
    output_path::String
)
    # Extract axes (x = nmemory, y = nsteps)
    xvals = sort(unique(k[2] for k in keys(results)))  # nmemory
    yvals = sort(unique(k[1] for k in keys(results)))  # nsteps
    
    # Build matrix
    Z = fill(NaN, length(yvals), length(xvals))
    for ((s, m), v) in results
        xi = findfirst(==(m), xvals)
        yi = findfirst(==(s), yvals)
        if xi !== nothing && yi !== nothing
            Z[yi, xi] = v
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
        ["\$\\mathrm{$(@sprintf("%.2f", value))}\$" for value in colorbar_tick_values],
        ","
    )
    
    colorbar_style =
        "{ytick={" * colorbar_yticks * "}," *
        "yticklabels={" * colorbar_yticklabels * "}," *
        "yticklabel style={font=\\fontsize{" *
        string(TICK_FONT_SIZE) * "}{" *
        string(TICK_FONT_SIZE + 7) *
        "}\\selectfont}," *
        "title={\$\\mathrm{RMSE}\$}," *
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
        title = feature_title(var_name, peak_type),
        titlefont = font(TITLE_FONT_SIZE),
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
        top_margin = 8mm,
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
    
    # Cell values (white text, no rotation)
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
    println("    Saved: $(basename(output_path))")
    
    return best_rmse, best_x, best_y
end

# -------------------- MAIN --------------------

function main()
    println("=" ^ 60)
    println("PGFPlotsX Paper-style Peak Heatmap Generation")
    println("Config: $FILTER_CONFIG")
    println("=" ^ 60)
    
    for peak_type in PEAK_TYPES
        println("\n--- $peak_type ---")
        
        for var in VARIABLES
            results_file = joinpath("results_peak", peak_type, var, "results.jld2")
            
            if !isfile(results_file)
                println("  Skipping $var: no results file")
                continue
            end
            
            println("  Processing: $var")
            results = load(results_file, "results")
            
            # Generate heatmap (naming format from plot_heatmap_maplike_paper.jl)
            which_str = peak_type == "localmin" ? "min" : "max"
            output_path = joinpath(OUTPUT_DIR, "heatmap_maplike_open_rmse_mem_vs_nt_which_$(which_str)_target_value_feature_$(var).png")
            try
                best_rmse, best_x, best_y = generate_peak_heatmap(
                    results, var, peak_type, output_path
                )
                println("    Best: Steps=$best_y, Memory=$best_x, RMSE=$(round(best_rmse, digits=4))")
            catch e
                println("    Error: $e")
            end
        end
    end
    
    println("\n" * "=" ^ 60)
    println("Done! Output: $OUTPUT_DIR")
    println("=" ^ 60)
end

main()
