push!(
    LOAD_PATH,
    normpath(joinpath(@__DIR__, "..", "package", "QuantumCircuits_demo", "src")),
    normpath(joinpath(@__DIR__, "..", "package", "VQC_demo", "src")),
)
using QuantumCircuits, QuantumCircuits.Gates
using VQC, VQC.Utilities
using Flux:train!
using Flux
using Random
using Statistics
using StatsBase
using LinearAlgebra
using SparseArrays


include("auxiliary.jl")
include("core.jl")
include("circuitQR.jl")
