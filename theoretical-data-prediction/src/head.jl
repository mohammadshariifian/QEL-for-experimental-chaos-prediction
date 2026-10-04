push!(LOAD_PATH,"../QuantumCircuits/src","../VQC/src")
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
