@everywhere using StatsBase
@everywhere using Distributions
@everywhere using Random
using Sobol
using PyCall
@pyimport MLsurrogate.Catboost as surro
# wrapper function to run Monte carlo simulations of the ABM
@everywhere function MonteCarlo(parameters::Array{Float64,2},montecarlos::Int, seedstart::Int, TIME::Int, transient::Int)
        N = size(parameters)[1]
        Npara = size(parameters)[2]
        results = Array{Float64,3}(undef,N,montecarlos,10)
        for i = 1 : N
            results[i,:,:] = sim(montecarlos,seedstart,parameters[i,:],TIME,transient)
        end
        return results
end

# function to compute the simulated joint moment estimator
# emp is a 2x10 dimensional array, where the first row contains the empirical moments and the second row contains the standard deviations
# data refers to the simulated data in terms of a three-dimensional array
# alpha is the critical value. for a two-tailed 95% confidence interval, alpha equals 1.96.
function momentsS(emp::Array{Float64,2},data::Array{Float64,3},alpha::Float64)
    N = size(data)[1]
    M = size(data)[2]
    emp1 = zeros(M,10)
    emp2 = zeros(M,10)
    alpha1 = fill(alpha,(M,10))
    for i = 1:M
        emp1[i,:] = emp[1,:]
        emp2[i,:] = emp[2,:]
    end
    jmcrS = zeros(N)
    for i= 1:N
        jmcrS[i] = count(sum((abs.(data[i,:,:]-emp1)./emp2).<alpha,dims=2).==10)
    end
    return jmcrS
end

# additional features to foster the performance of the surrogate model
# features depent on the structure of the ABM
@everywhere function featureAdd(data::Array{Float64,2})
        N = size(data)[1]
        addFeatures = zeros(N,4)
        addFeatures[:,1] = data[:,3]./data[:,2]
        addFeatures[:,2] = data[:,3]./data[:,2]./data[:,4]
        addFeatures[:,3] = data[:,5]./data[:,7]
        addFeatures[:,4] = data[:,5]./data[:,7]./data[:,6]
        return addFeatures
    end
@everywhere function esti16(min_Sobol::Array{Float64,1}, max_Sobol::Array{Float64,1}, alpha::Float64, emp_data::Array{Float64,2}, grid_size::Int, rounds::Int, round_samples::Int, montecarlos::Int, TIME::Int, transient::Int, MC::Int)
        # number of parameters
        n_dimensions= 9
        # number of simulations per cpu core
        sim_n = Int(floor(round_samples/16.0))
        # array to store results
        jmcr_results = Array{Float64,2}(undef,0,10)
        timings = zeros(MC)
        seedSurro = 1

        # initial grid search over 2500 parameter combinations
        # sampling via Sobol sequences
        initial_exploration_range = SobolSeq(n_dimensions, min_Sobol, max_Sobol)
        skip(initial_exploration_range,2500)
        train_set_X= zeros(2500,n_dimensions)
        for i = 1:2500 train_set_X[i,:]=next!(initial_exploration_range) end
        sim_n2 = Int(floor(2500/16.0))

        for j = 1:MC
            timings[j] = @elapsed begin
                        p1=@spawn MonteCarlo(train_set_X[1:sim_n2,:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p2=@spawn MonteCarlo(train_set_X[(sim_n2+1):(sim_n2*2),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p3=@spawn MonteCarlo(train_set_X[(sim_n2*2+1):(sim_n2*3),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p4=@spawn MonteCarlo(train_set_X[(sim_n2*3+1):(sim_n2*4),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p5=@spawn MonteCarlo(train_set_X[(sim_n2*4+1):(sim_n2*5),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p6=@spawn MonteCarlo(train_set_X[(sim_n2*5+1):(sim_n2*6),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p7=@spawn MonteCarlo(train_set_X[(sim_n2*6+1):(sim_n2*7),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p8=@spawn MonteCarlo(train_set_X[(sim_n2*7+1):(sim_n2*8),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p9=@spawn MonteCarlo(train_set_X[(sim_n2*8+1):(sim_n2*9),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p10=@spawn MonteCarlo(train_set_X[(sim_n2*9+1):(sim_n2*10),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p11=@spawn MonteCarlo(train_set_X[(sim_n2*10+1):(sim_n2*11),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p12=@spawn MonteCarlo(train_set_X[(sim_n2*11+1):(sim_n2*12),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p13=@spawn MonteCarlo(train_set_X[(sim_n2*12+1):(sim_n2*13),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p14=@spawn MonteCarlo(train_set_X[(sim_n2*13+1):(sim_n2*14),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p15=@spawn MonteCarlo(train_set_X[(sim_n2*14+1):(sim_n2*15),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                        p16=@spawn MonteCarlo(train_set_X[(sim_n2*15+1):(2500),:],montecarlos,((j-1)*montecarlos),TIME,transient);

                        p1 = fetch(p1)
                        p2 = fetch(p2)
                        p3 = fetch(p3)
                        p4 = fetch(p4)
                        p5 = fetch(p5)
                        p6 = fetch(p6)
                        p7 = fetch(p7)
                        p8 = fetch(p8)
                        p9 = fetch(p9)
                        p10 = fetch(p10)
                        p11 = fetch(p11)
                        p12 = fetch(p12)
                        p13 = fetch(p13)
                        p14 = fetch(p14)
                        p15 = fetch(p15)
                        p16 = fetch(p16)
                        Results = vcat(p1, p2, p3, p4, p5, p6, p7 ,p8, p9, p10, p11, p12, p13, p14, p15, p16)

                        #compute joint moment estimator for initial grid run
                        jmcr = momentsS(emp_data[(j*2-1):(j*2),:],Results,alpha)
                        dataset = hcat(train_set_X,jmcr)

                        # initialize sampling pool for main estimation
                        abm_exploration_range = SobolSeq(n_dimensions, min_Sobol, max_Sobol)
                        skip(abm_exploration_range,grid_size)
                        unevaluated_set_X= zeros(grid_size,n_dimensions)
                        for i = 1:grid_size unevaluated_set_X[i,:]=next!(abm_exploration_range) end

                        # run estimation over k rounds
                        for k= 1:rounds
                                                # call Python function to train surrogate model
                                                surrogates = PyObject([])
                                                surrogates = surro.fit_surrogate_model(dataset,featureAdd(dataset),seedSurro)
                                                seedSurro += 1
                                                # make predictions
                                                predictions = convert(Array{Float64,1},surrogates[:predict](hcat(unevaluated_set_X,featureAdd(unevaluated_set_X))))
                                                predictions_selections = sortperm(predictions, rev=true)[1:round_samples]
                                                round_set_X = unevaluated_set_X[predictions_selections,:]
                                                # update unevaluated_set_X = delete latest round selections
                                                unevaluated_set_X = unevaluated_set_X[setdiff(1:end,predictions_selections),:]
                                                # Run true ABM in parallel over 16 cores
                                                p1=@spawn MonteCarlo(round_set_X[1:sim_n,:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p2=@spawn MonteCarlo(round_set_X[(sim_n+1):(sim_n*2),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p3=@spawn MonteCarlo(round_set_X[(sim_n*2+1):(sim_n*3),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p4=@spawn MonteCarlo(round_set_X[(sim_n*3+1):(sim_n*4),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p5=@spawn MonteCarlo(round_set_X[(sim_n*4+1):(sim_n*5),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p6=@spawn MonteCarlo(round_set_X[(sim_n*5+1):(sim_n*6),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p7=@spawn MonteCarlo(round_set_X[(sim_n*6+1):(sim_n*7),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p8=@spawn MonteCarlo(round_set_X[(sim_n*7+1):(sim_n*8),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p9=@spawn MonteCarlo(round_set_X[(sim_n*8+1):(sim_n*9),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p10=@spawn MonteCarlo(round_set_X[(sim_n*9+1):(sim_n*10),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p11=@spawn MonteCarlo(round_set_X[(sim_n*10+1):(sim_n*11),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p12=@spawn MonteCarlo(round_set_X[(sim_n*11+1):(sim_n*12),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p13=@spawn MonteCarlo(round_set_X[(sim_n*12+1):(sim_n*13),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p14=@spawn MonteCarlo(round_set_X[(sim_n*13+1):(sim_n*14),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p15=@spawn MonteCarlo(round_set_X[(sim_n*14+1):(sim_n*15),:],montecarlos,((j-1)*montecarlos),TIME,transient);
                                                p16=@spawn MonteCarlo(round_set_X[(sim_n*15+1):(round_samples),:],montecarlos,((j-1)*montecarlos),TIME,transient);

                                                p1 = fetch(p1)
                                                p2 = fetch(p2)
                                                p3 = fetch(p3)
                                                p4 = fetch(p4)
                                                p5 = fetch(p5)
                                                p6 = fetch(p6)
                                                p7 = fetch(p7)
                                                p8 = fetch(p8)
                                                p9 = fetch(p9)
                                                p10 = fetch(p10)
                                                p11 = fetch(p11)
                                                p12 = fetch(p12)
                                                p13 = fetch(p13)
                                                p14 = fetch(p14)
                                                p15 = fetch(p15)
                                                p16 = fetch(p16)
                                                Results = vcat(p1, p2, p3, p4, p5, p6, p7 ,p8, p9, p10, p11, p12, p13, p14, p15, p16)

                                                # compute joint moment estimator
                                                round_set_y = momentsS(emp_data[(j*2-1):(j*2),:],Results,alpha)
                                                # update dataset
                                                dataset = vcat(dataset,hcat(round_set_X,round_set_y))

                                                println("MC run: ",j)
                                                println("Round ", k," finished. Num of new positives: ",length(findall(round_set_y.>0)) ," Best score: ",maximum(dataset[:,10]))
                        end
                        jmcr_results = vcat(jmcr_results,dataset[findall(dataset[:,10].==maximum(dataset[:,10])),:])
                    end

        end
        return jmcr_results, timings
end
