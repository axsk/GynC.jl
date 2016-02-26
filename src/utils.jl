using Distributions
import Distributions: logpdf, minimum, maximum, length, insupport, _logpdf, size


type UnivariateDensityDistribution <: ContinuousUnivariateDistribution
  lpdf::Function
  min::Real
  max::Real
end

type MultivariateDensityDistribution <: ContinuousMultivariateDistribution
  lpdf::Function
  dim
  insupport::Function
end

myisapprox(x::Number, y::Number) = (isapprox(x,y) || isequal(x,y))
myisapprox(x,y) = (r=map(myisapprox, x,y); all(r))

""" return memoized version of f, caching the last n calls' results (overwriting on same-argument calls) """
function cache(fn::Function, n::Int)
    cin  = fill!(Vector{Any}(n), nothing)
    cout = fill!(Vector{Any}(n), nothing)
    c = 1
    function (args...)
        i = any(cin.==nothing) ? 0 : findfirst(cargs->myisapprox(cargs,args), cin)
        res = i == 0 ? fn(args...) : cout[i]
        cin[c] = deepcopy(args)
        cout[c] = res
        c = c % n + 1
        res
    end
end

""" Constructs a Distribution based on the given density function """
DensityDistribution(pdf::Function; kwargs...) = DensityDistribution(1, pdf; kwargs...)

function DensityDistribution(dim, pdf::Function; log=false, insupport::Function=((x)->true), intervall=[-Inf,Inf])
  lpdf = log ? pdf : x -> log(pdf(x)) 
  (isa(dim, Number) && dim <= 1) ? 
    UnivariateDensityDistribution(lpdf, intervall[1], intervall[2]) : 
    MultivariateDensityDistribution(lpdf, dim, insupport)
end
  
logpdf(d::UnivariateDensityDistribution, x::Real)        = d.lpdf(x)
minimum(d::UnivariateDensityDistribution)                = d.min
maximum(d::UnivariateDensityDistribution)                = d.max

logpdf(d::MultivariateDensityDistribution, x::DenseMatrix, transform::Bool=true) = d.lpdf(x)
#_logpdf(d::MultivariateDensityDistribution, x::DenseMatrix)   = d.lpdf(x)
insupport(d::MultivariateDensityDistribution, x::DenseMatrix) = d.insupport(x)
length(d::MultivariateDensityDistribution)               = d.dim
size(d::MultivariateDensityDistribution)                 = d.dim

""" Given samplings (of the same size), concatenate them to form their mean sampling """ 
function samplemean(chains::Vector{Mamba.ModelChains}) 
  for i = 1:length(chains)-1
    size(chains[i], 1) == size(chains[i+1], 1) ||
      warn("concatenated chains have not same length")
  end
  Chains(cat(1, [c.value for c in chains]...))
end


const fortranpath = joinpath(dirname(@__FILE__),"..","fortran","GynC.so") 

""" Wrapper to the LIMEX solver for the GynC model 
solve the model for the times t given initial condition y0 and parameters parms, and store the result in y"""
function gync(y0::Vector{Float64}, tspan::Vector{Float64}, Parms::Vector{Float64})
  n = length(y0)
  m = length(tspan)
  y = Array{Float64}(n,m)

  ccall((:limstep_, fortranpath), Ptr{Array{Float64,2}},
    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}),
    y, copy(y0), tspan, &n, &m, Parms)
  y
end

function mlegync(tspan::Vector{Float64}=collect(1:31.))
  parms, y0 = loadmles()
  gync(y0, tspan, parms)
end

### MergedChain

""" memory efficient structure to represent the merged chain """
type MergedChain{T<:Real} <: AbstractMatrix{T}
  chains::Vector{Matrix{T}}

  function MergedChain{T}(chains::Vector{Matrix{T}})
    all([size(c) for c in chains] .== size(chains[1])) || warn("chains dont have same size")
    new(chains)
  end
end

mergedchain(chains...) = MergedChain{Float64}(Vector{Matrix{Float64}}(chains...))

chainlength(mc::MergedChain) = size(mc.chains[1], 1)
nchains(mc::MergedChain) = length(mc.chains)

Base.size(mc::MergedChain) = (nchains(mc) * chainlength(mc), size(mc.chains[1], 2))

function Base.getindex(mc::MergedChain, i::Int, j::Int)
  chain = floor((i-1) / chainlength(mc)) + 1 |> Int
  index = ((i-1) % chainlength(mc) + 1)
  mc.chains[chain][index, j]
end
