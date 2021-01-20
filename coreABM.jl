@everywhere using StatsBase
@everywhere using Distributions
@everywhere using Random

# function to compute hill tail indices at 2.5% and 5%
@everywhere function hill(x::Vector{Float64})
    x=log.(sort(abs.(x),rev=true))
    z2 = Int(round(0.05*length(x)))
    z1 = Int(round(z2/2))
    return [1/(1/z1*(sum(x[1:z1])-z1*x[z1+1])),1/(1/z2*(sum(x[1:z2])-z2*x[z2+1]))]
end
# model code of Schmitt et al. (2020)
@everywhere function sim(seed::Int, seedstart::Int, parameter::Array{Float64,1},TIME::Int,transient::Int) #core ABM model
    nt=100;
    a=1.0;
    F=0.;
    m=parameter[1];
    dh=parameter[2];
    dl=parameter[3];
    ds=200000.0;
    vbar=parameter[4];
    rhol=parameter[7];
    rhoh=parameter[5];
    rhos=200000.0;
    cbar=parameter[6];
    b=parameter[8];
    c=parameter[9];
    results0 = zeros(1,seed,10) # monte carlo inner loop results
    for j = 1 : seed
        LogP=zeros(TIME);
        Rho=zeros(TIME);
        vol2=vbar;
        Random.seed!(seedstart+j);
        for i = 4 : TIME
            vol1=m*vol2+(1-m)*(LogP[i-1]-LogP[i-2])^2;
            Rho[i]=(rhol+(rhoh-rhol)/(1+exp(-rhos*(((LogP[i-1]-LogP[i-2])-(LogP[i-2]-LogP[i-3]))^2-cbar))));
            LogP[i]=LogP[i-1]+a*(nt*(b*(LogP[i-1]-LogP[i-2])+c*(F-LogP[i-1]))+sqrt(dl+(dh-dl)/(1+exp(-ds*(vol1-vbar))))*sqrt(nt+nt*(nt-1)*(Rho[i]))*randn());
            vol2=vol1;
        end
        # log returns
        ret = (LogP[transient+1:end]-LogP[transient:(end-1)])*100.0
        # compute moments from simulated data, i.e. log returns
        results0[1,j,:] = vcat(mean(abs.(ret)),hill(ret),autocor(ret,[1])[1],getindex(autocor(abs.(ret),1:100),[3,6,12,25,50,100]))
    end
    return results0
end
