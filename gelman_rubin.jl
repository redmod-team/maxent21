using DelimitedFiles
# using Plots
using Statistics

fn = expanduser(
    "~/Downloads/diatom_17/INPUT_DATA/TMP/MCMC_V2_1997-2001/mc_out.dat")
data = readdlm(fn, skipstart=1)

## See https://bookdown.org/rdpeng/advstatcomp/monitoring-convergence.html

L = 1000
J = 1000
xj = Array{Float64,2}(undef, J, 10)
sj = Array{Float64,2}(undef, J, 10)
for k in 1:J
    xj[k, :] = mean(data[1+(k-1)*1000:(k-1)*1000 + L,:], dims=1)
    sj[k, :] = var(data[1+(k-1)*1000:(k-1)*1000 + L,:], dims=1)
end


X = mean(xj, dims=1)
B = var(xj, dims=1)
W = sum(sj, dims=1)/J

R = ((L-1)/L*W + B/L)./W
