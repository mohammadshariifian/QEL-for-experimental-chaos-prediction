function matlab_findpeaks_auto(x::Vector{Float64}; mode::Symbol, pmin::Float64)
    if !USE_MATLAB_PEAKS
        error("MATLAB peak detection is disabled. Set MAPLIKE=false or replace peak detection with Julia code.")
    end
    MATLAB.put_variable(:x, x)
    MATLAB.put_variable(:pmin, float(pmin))

    if mode === :max
        mat"""
        x = x(:);
        [~, locs1, w1, ~] = findpeaks(x, 'MinPeakDistance', 1, 'MinPeakProminence', pmin);
        """
    elseif mode === :min
        mat"""
        x = x(:);
        [~, locs1, w1, ~] = findpeaks(-x, 'MinPeakDistance', 1, 'MinPeakProminence', pmin);
        """
    else
        error("mode must be :max or :min")
    end

    locs1 = Int.(round.(vec(MATLAB.get_variable(:locs1))))
    w1    = vec(MATLAB.get_variable(:w1))

    isempty(locs1) && return Int[]

    d_auto = max(1, round(Int, 1.25 * median(w1)))
    MATLAB.put_variable(:dauto, float(d_auto))

    if mode === :max
        mat"""
        x = x(:);
        [~, locs2] = findpeaks(x, 'MinPeakDistance', dauto, 'MinPeakProminence', pmin);
        """
    else
        mat"""
        x = x(:);
        [~, locs2] = findpeaks(-x, 'MinPeakDistance', dauto, 'MinPeakProminence', pmin);
        """
    end

    return Int.(round.(vec(MATLAB.get_variable(:locs2))))
end

function extrema_indices_from_normalized(
    x::Vector{Float64};
    which::Symbol = :max,
    feature_name::String = "",
)
    if !USE_MATLAB_PEAKS
        error("MATLAB peak detection is disabled. Set MAPLIKE=false or replace peak detection with Julia code.")
    end
    if haskey(PEAK_SETTINGS, feature_name)
        s = PEAK_SETTINGS[feature_name]
        MATLAB.put_variable(:x, x)

        if which === :max
            MATLAB.put_variable(:pmin, float(s.max_prominence))
            MATLAB.put_variable(:dmin, float(s.max_distance))
            mat"""
            x = x(:);
            [~, locs] = findpeaks(x, ...
                'MinPeakProminence', pmin, ...
                'MinPeakDistance', dmin);
            """
        elseif which === :min
            MATLAB.put_variable(:pmin, float(s.min_prominence))
            MATLAB.put_variable(:dmin, float(s.min_distance))
            mat"""
            x = x(:);
            [~, locs] = findpeaks(-x, ...
                'MinPeakProminence', pmin, ...
                'MinPeakDistance', dmin);
            """
        else
            error("which must be :max or :min for fixed-parameter features")
        end

        return Int.(round.(vec(MATLAB.get_variable(:locs))))
    end

    # --- DEFAULT (existing behavior for other features) ---
    pmin = auto_prominence_from_signal(x)

    if which === :max
        return matlab_findpeaks_auto(x; mode=:max, pmin=pmin)
    elseif which === :min
        return matlab_findpeaks_auto(x; mode=:min, pmin=pmin)
    elseif which === :both
        idx1 = matlab_findpeaks_auto(x; mode=:max, pmin=pmin)
        idx2 = matlab_findpeaks_auto(x; mode=:min, pmin=pmin)
        return sort!(unique!(vcat(idx1, idx2)))
    else
        error("which must be :max, :min, or :both")
    end
end
