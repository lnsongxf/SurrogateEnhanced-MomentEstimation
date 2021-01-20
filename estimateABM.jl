using Distributed
if nprocs() == 1
    addprocs(Sys.CPU_THREADS)
end
using CSV
using DataFrames
@everywhere include("coreABM.jl")
@everywhere include("estimationRoutine.jl")

# emp_data contains moments + standard dev of 100 independently (pre-) generated pseudo-empirical data
emp_data = convert(Array{Float64},CSV.read("momentsGrid_PseudoEmp_20k.csv",header=true))
# lower and upper bounds for Sobol sampling
min_Sobol = convert(Matrix{Float64},CSV.read("min1.csv",header=false))[:,1]
max_Sobol = convert(Matrix{Float64},CSV.read("max1.csv", header=false))[:,1]

# run the estimation
# estimation results will be stored in resultsS, timing results in resultsT
# function esti16 needs the following input: lower bound for Sobol sampling, upper bound for Sobol sampling,
# critical value, empirical moments + standard dev, pool size, number of rounds, number of simulation runs per round,
# simulated time length, transient phase, number of repetitions
resultsS, resultsT = esti16(min_Sobol,max_Sobol,1.96,emp_data,2000000,4,400,400,20100,100,100)
