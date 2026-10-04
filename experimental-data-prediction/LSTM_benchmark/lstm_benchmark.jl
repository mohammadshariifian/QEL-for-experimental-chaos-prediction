using Base.Threads
using CSV, DataFrames, JSON, Dates
using DelimitedFiles
using Pkg
const PROJECT_ROOT = @__DIR__
Pkg.activate(PROJECT_ROOT)

const DATA_MODE = :experiment
# Valid values: :theory or :experiment

const THEORY_PAST_INTERVALS = [21, 7, 34, 68]
const EXPERIMENT_PAST_INTERVALS = [25, 13, 31, 27]

const PAST_INTERVALS = if DATA_MODE === :theory
    THEORY_PAST_INTERVALS
elseif DATA_MODE === :experiment
    EXPERIMENT_PAST_INTERVALS
else
    error("Unknown DATA_MODE=$DATA_MODE")
end

const THEORY_DATA_FILE = abspath(joinpath(
    @__DIR__, "..",  "data", "output_320.csv"
))

const EXPERIMENT_DATA_FILE = abspath(joinpath(
    @__DIR__, "..",  "data",
    "board_1_run_61_new_lowpass.csv"
))

const EXPERIMENT_FEATURES = ["u1", "u2", "iB", "iC"]

const FEATURE_NAMES = if DATA_MODE === :experiment
    EXPERIMENT_FEATURES
else
    ["var1", "var2", "var3", "var4"]
end

const EXPERIMENT_T_CTX = (5.0, 210.0)   # μs
const EXPERIMENT_T_FUT = (210.0, 300.0) # μs



const PYTHON_EXE = let
    # Highest priority: explicit user-provided path.
    if haskey(ENV, "LSTM_PYTHON") && !isempty(ENV["LSTM_PYTHON"])
        expanduser(ENV["LSTM_PYTHON"])

    # Use Python from the currently activated Conda environment.
    elseif haskey(ENV, "CONDA_PREFIX") && !isempty(ENV["CONDA_PREFIX"])
        if Sys.iswindows()
            joinpath(ENV["CONDA_PREFIX"], "python.exe")
        else
            joinpath(ENV["CONDA_PREFIX"], "bin", "python")
        end

    # Fall back to Python available on PATH.
    else
        python_path = Sys.which(Sys.iswindows() ? "python.exe" : "python3")

        if python_path === nothing
            python_path = Sys.which("python")
        end

        python_path === nothing && error(
            "No Python executable found. Activate the required Conda environment " *
            "or set the LSTM_PYTHON environment variable."
        )

        python_path
    end
end

const OUTPUT_DIR = get(
    ENV,
    "QEL_OUTPUT_DIR",
    normpath(joinpath(PROJECT_ROOT, "..", "output")),
)

const ACTIVE_DATA_CONFIG = Ref{Any}(nothing)

data_output_root() = joinpath(
    OUTPUT_DIR,
    String(DATA_MODE),
)

result_dir(mode::Int, output_name::String) = joinpath(
    data_output_root(),
    "$(mode)to1",
    output_name,
)

const PYTHON_SCRIPT_MULTI = joinpath(@__DIR__, "lstm_single_multi.py")

const N_VARS = length(FEATURE_NAMES)
const VAR_IDS = Dict(i => FEATURE_NAMES[i] for i in 1:N_VARS)
const LOOKBACKS = collect(1:2)
const HIDDEN_UNITS = collect(1:14)
const N_LAYERS = collect(1:14)
const SEEDS = [42, 43, 44, 45, 46]

const INPUT_IDS_3TO1 = [1, 3, 4]
const INPUT_IDS_4TO1 = [1, 2, 3, 4]

mode_arg = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
output_id_arg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

function read_board_feature_data(
    path::AbstractString,
    input_features::Vector{String},
)
    df = CSV.read(
        path,
        DataFrame;
        delim='\t',
        header=false,
        comment="*",
        quotechar='\0',
        ignorerepeated=true,
        types=Float64,
    )

    rename!(df, [:t_s, :u1, :u2, :u3, :iB, :iC])

    t_us_all = Vector{Float64}(df.t_s .* 1e6)

    X_all = permutedims(
        hcat([
            Vector{Float64}(df[!, Symbol(feature)])
            for feature in input_features
        ]...)
    )

    return t_us_all, Matrix{Float64}(X_all)
end

function prepare_benchmark_data()
    if DATA_MODE === :theory
        isfile(THEORY_DATA_FILE) || error(
            "Theory data file not found: $THEORY_DATA_FILE"
        )

        return (
            source_file=THEORY_DATA_FILE,
            data_file=THEORY_DATA_FILE,
            explicit_split=true,
            train_start_idx=1563,
            train_end_idx=65625,
            test_start_idx=65626,
            test_end_idx=93750,
        )

    elseif DATA_MODE === :experiment
        isfile(EXPERIMENT_DATA_FILE) || error(
            "Experimental data file not found: " *
            EXPERIMENT_DATA_FILE
        )

        t_us, X = read_board_feature_data(
            EXPERIMENT_DATA_FILE,
            EXPERIMENT_FEATURES,
        )

        size(X, 1) == N_VARS || error(
            "Expected $N_VARS experimental features, " *
            "received $(size(X, 1))."
        )

        issorted(t_us) || error(
            "Experimental time vector is not sorted."
        )

        prepared_dir = joinpath(
            data_output_root(),
            "_prepared_input",
        )

        mkpath(prepared_dir)

        prepared_file = joinpath(
            prepared_dir,
            "board_1_run_61_u1_u2_iB_iC.csv",
        )

        # Python expects one variable per row and time across columns.
        writedlm(prepared_file, X, ',')

        # Convert Julia's 1-based indices to Python's 0-based indices.
        # Training: 5 <= t < 210 μs
        train_start_idx =
            searchsortedfirst(t_us, EXPERIMENT_T_CTX[1]) - 1

        train_end_idx =
            searchsortedfirst(t_us, EXPERIMENT_T_CTX[2]) - 2

        # Testing: 210 <= t <= 300 μs
        test_start_idx =
            searchsortedfirst(t_us, EXPERIMENT_T_FUT[1]) - 1

        test_end_idx =
            searchsortedlast(t_us, EXPERIMENT_T_FUT[2]) - 1

        train_start_idx <= train_end_idx || error(
            "Invalid experimental training interval."
        )

        test_start_idx <= test_end_idx || error(
            "Invalid experimental testing interval."
        )

        return (
            source_file=EXPERIMENT_DATA_FILE,
            data_file=prepared_file,
            explicit_split=true,
            train_start_idx=train_start_idx,
            train_end_idx=train_end_idx,
            test_start_idx=test_start_idx,
            test_end_idx=test_end_idx,
        )

    else
        error(
            "Unknown DATA_MODE=$DATA_MODE. " *
            "Use :theory or :experiment."
        )
    end
end

function get_output_ids(mode::Int, output_id::Int)
    if mode == 3
        return output_id == 0 ? INPUT_IDS_3TO1 : [output_id]

    elseif mode == 4
        return output_id == 0 ? collect(1:N_VARS) : [output_id]

    else
        error(
            "Unknown mode: $mode. " *
            "Only modes 3 and 4 are supported."
        )
    end
end



function generate_configs(mode::Int, output_id::Int)
    configs = []
    for seed in SEEDS, nl in N_LAYERS, hu in HIDDEN_UNITS, lb in LOOKBACKS
        priority = (seed == 42 ? 0 : 100) + (nl == 1 ? 0 : 50) + findfirst(==(hu), HIDDEN_UNITS) * 10 + findfirst(==(lb), LOOKBACKS)
        push!(configs, (mode=mode, output_id=output_id, lookback=lb, hidden=hu, n_layers=nl, seed=seed, priority=priority))
    end
    sort!(configs, by=c -> c.priority)
    return configs
end

function run_one(cfg)
    mode = cfg.mode
    output_id = cfg.output_id
    output_name = VAR_IDS[output_id]

    out_dir = result_dir(mode, output_name)

    python_script = PYTHON_SCRIPT_MULTI
    
    mkpath(out_dir)
    ds = PAST_INTERVALS[output_id]

    result_file = joinpath(
        out_dir,
        "result_$(output_name)_DS$(ds)_L$(cfg.lookback)_HU$(cfg.hidden)_NL$(cfg.n_layers)_S$(cfg.seed).json"
    )

    if isfile(result_file)
        return JSON.parsefile(result_file)
    end

    data_cfg = ACTIVE_DATA_CONFIG[]

    cfg_dict = Dict{String,Any}(
        "mode" => mode,
        "output_id" => output_id,
        "lookback" => cfg.lookback,
        "hidden" => cfg.hidden,
        "n_layers" => cfg.n_layers,
        "seed" => cfg.seed,
        "output_dir" => out_dir,
        "n_vars" => N_VARS,
        "var_names" => FEATURE_NAMES,
        "data_mode" => String(DATA_MODE),
        "data_file" => data_cfg.data_file,
        "past_intervals" => PAST_INTERVALS,
    )

    if data_cfg.explicit_split
        cfg_dict["train_start_idx"] =
            data_cfg.train_start_idx

        cfg_dict["train_end_idx"] =
            data_cfg.train_end_idx

        cfg_dict["test_start_idx"] =
            data_cfg.test_start_idx

        cfg_dict["test_end_idx"] =
            data_cfg.test_end_idx
    end

    cfg_json = JSON.json(cfg_dict)

    cmd = `$PYTHON_EXE $python_script $cfg_json`
    output = read(cmd, String)
    result = JSON.parse(output)
    return result
end

function run_output(mode::Int, output_id::Int)
    output_name = VAR_IDS[output_id]
    mode_str = "$(mode)to1"
    println("[$(now())] === Processing $mode_str -> $output_name ===")

    configs = generate_configs(mode, output_id)
    println("[$(now())] Total configs: $(length(configs))")

    results_lock = ReentrantLock()
    results = []
    completed = Atomic{Int}(0)
    failed = Atomic{Int}(0)
    @threads for cfg in configs
        try
            result = run_one(cfg)
            lock(results_lock) do
                push!(results, result)
            end
            n = atomic_add!(completed, 1) + 1
            if n % 10 == 0
                println("[$(now())] [$mode_str->$output_name] $n/$(length(configs)) completed | L=$(cfg.lookback) HU=$(cfg.hidden) NL=$(cfg.n_layers) S=$(cfg.seed): LSTM RMSE=$(round(result["lstm_rmse"], sigdigits=4))")
            end
        catch e
            atomic_add!(failed, 1)
            println(
                "[$(now())] ERROR: " *
                "L=$(cfg.lookback) HU=$(cfg.hidden) " *
                "NL=$(cfg.n_layers) S=$(cfg.seed): $e"
            )
        end
    end

    n_failed = failed[]

    if n_failed > 0
        @warn "$n_failed configurations failed. The summary contains only $(length(results)) successful configurations."
    end

    if isempty(results)
        error(
            "All $(length(configs)) configurations failed for " *
            "$mode_str -> $output_name. No summary was written."
        )
    end

    out_dir = result_dir(mode, output_name)

    mkpath(out_dir)

    df = DataFrame(results)

    sort!(
        df,
        [:lookback, :hidden, :n_layers, :seed],
    )

    csv_file = joinpath(
        out_dir,
        "summary_metrics.csv",
    )
    CSV.write(csv_file, df)

    println("[$(now())] Saved: $csv_file")

    return df
end

function main()
    println("Threads: $(nthreads())")
    println("Mode: $(mode_arg)to1")
    println("Python executable: $PYTHON_EXE")
    println("Output directory: $OUTPUT_DIR")

    mode_arg in (3, 4) || error(
    "Invalid mode $mode_arg. Valid modes are 3 and 4."
)

    output_id_arg in 0:N_VARS || error(
        "Invalid output ID $output_id_arg. Use 0 for all variables or 1:$N_VARS."
    )
    if mode_arg == 3 &&
    output_id_arg != 0 &&
    !(output_id_arg in INPUT_IDS_3TO1)

        error(
            "For 3-to-1 mode, output ID must be " *
            "0, 1, 3, or 4."
        )
    end

    isfile(PYTHON_EXE) || error(
        "Python executable not found: $PYTHON_EXE"
    )


    isfile(PYTHON_SCRIPT_MULTI) || error(
        "Python script not found: $PYTHON_SCRIPT_MULTI"
    )

    try
        run(`$PYTHON_EXE -c "import torch, numpy"`)
    catch
        error(
            "The selected Python cannot import torch and numpy.\n" *
            "Python executable: $PYTHON_EXE"
        )
    end

    mkpath(OUTPUT_DIR)

    ACTIVE_DATA_CONFIG[] = prepare_benchmark_data()
    data_cfg = ACTIVE_DATA_CONFIG[]

    println("Data mode: $DATA_MODE")
    println("Source data file: $(data_cfg.source_file)")
    println("Python input file: $(data_cfg.data_file)")
    println("Features: $FEATURE_NAMES")

    if data_cfg.explicit_split
        println(
            "Training indices: " *
            "$(data_cfg.train_start_idx):" *
            "$(data_cfg.train_end_idx)"
        )

        println(
            "Testing indices: " *
            "$(data_cfg.test_start_idx):" *
            "$(data_cfg.test_end_idx)"
        )
    end

    output_ids = get_output_ids(mode_arg, output_id_arg)
    println("Output variables: $([VAR_IDS[i] for i in output_ids])")

    for oid in output_ids
        run_output(mode_arg, oid)
    end

    all_dfs = DataFrame[]

    for oid in 1:N_VARS
        output_name = VAR_IDS[oid]

        csv_file = joinpath(
            result_dir(mode_arg, output_name),
            "summary_metrics.csv",
        )

        if isfile(csv_file)
            push!(
                all_dfs,
                CSV.read(csv_file, DataFrame)
            )
        end
    end

    if !isempty(all_dfs)
        combined = vcat(all_dfs...)

        combined_file = joinpath(
            data_output_root(),
            "$(mode_arg)to1_summary_all_$(String(DATA_MODE)).csv",
        )

        CSV.write(combined_file, combined)

        println(
            "[$(now())] Saved combined: $combined_file"
        )
    end
end

main()
