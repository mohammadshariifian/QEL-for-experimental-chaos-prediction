using DifferentialEquations, DelimitedFiles

# Parameters
VS   = 5.0
R    = 1000.0
L1   = 15e-6
L2   = 220e-6
C1   = 220e-12
C2   = 5e-12
Vth  = 0.6
beta = 200.0

# Initial conditions
uC1_0 = Vth
uC2_0 = Vth
iL1_0 = Vth / R
iL2_0 = Vth / R
y0 = [uC1_0; uC2_0; iL1_0; iL2_0]

# Time span: 0 to 100μs
tmax = 10e-5
dt = 32e-6 / 10000
tspan = (0.0, tmax)

function circuitODE!(dydt, y, p, t)
    uC1, uC2, iL1, iL2 = y
    Gamma = max(iL1, 0.0)
    iT = beta * Gamma * tanh(uC2 / (2 * Vth))
    dydt[1] = (VS - uC1) / (R * C1) - (iL1 + iL2) / C1
    dydt[2] = (iL2 - iT) / C2
    dydt[3] = (uC1 - Vth) / L1
    dydt[4] = (uC1 - uC2) / L2
end

prob = ODEProblem(circuitODE!, y0, tspan)
sol = solve(prob, Tsit5(), saveat=dt, abstol=1e-6, reltol=1e-6)

# Extract and transpose: 4 rows × N columns
data = Array(sol)  # 4 × N
open("output_320.csv", "w") do io
    for row in 1:4
        vals = [string(data[row, col]) for col in 1:size(data, 2)]
        println(io, join(vals, ","))
    end
end

println("Saved output_320.csv")
println("Points: $(size(data, 2))")
