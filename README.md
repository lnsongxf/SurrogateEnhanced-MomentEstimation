# SurrogateEnhanced-MomentEstimation
Julia code for Schwartz (2021)


The file <b>coreABM.jl</b> contains the code for simulating the agent-based model of Schmitt et al. (2020).

The main algorithm for the estimation is in the file <b>estimationRountine.jl</b>

<b>MLsurrogate</b> is a small Python library that uses the Catboost package to train a surrogate model. To run it from Julia using PyCall, it has to be saved in one of Python's path directories (sys.path).

















REFERENCES

Schmitt, N., Schwartz, I. and Westerhoff, F. (2020): Heterogeneous speculators and stock market dynamics: a simple agent-based computational model. The European Journal of Finance, 1-20.
Schwartz, I. (2021): Estimation of agent-based models: Testing and applying a simulated joint moment approach. Working Paper.
