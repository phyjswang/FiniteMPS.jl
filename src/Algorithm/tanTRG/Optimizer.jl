"""
     mutable struct ImaginaryTimeStatus
          β::Float64           
          ∇E2::Float64
          W::Float64
          history::Vector{ImaginaryTimeStatus}
     end

The status of imaginary-time evolution at current step.

# Fields

     β: current imaginary time
     ∇E2: estimated ⟨∇E, ∇E⟩/N at current step, where N is the system size
     W::estimated spectral width of effective Hamiltonian
     history:: history of previous statuses

# Constructor

     ImaginaryTimeStatus(β::Real, ∇E2::Real, W::Real; history::Vector{ImaginaryTimeStatus} = ImaginaryTimeStatus[])
"""
mutable struct ImaginaryTimeStatus
     β::Float64           
     ∇E2::Float64
     W::Float64
     history::Vector{ImaginaryTimeStatus}
     function ImaginaryTimeStatus(β::Real, ∇E2::Real, W::Real; history::Vector{ImaginaryTimeStatus} = ImaginaryTimeStatus[])
          @assert β ≥ 0 
          W < 0 && @warn "Estimated spectral width W = $(W) < 0"
          ∇E2 < 0 && @warn "Estimated ⟨∇E, ∇e⟩ = $(∇E2) < 0"
          return new(convert(Float64, β), convert(Float64, ∇E2), convert(Float64, W), history)
     end
end

function show(io::IO, st::ImaginaryTimeStatus)
     return print(io, "ImaginaryTimeStatus(β=$(st.β), β^2⟨∇E, ∇e⟩=$(st.β^2 * st.∇E2), W=$(st.W))")
end

"""
     update!(st::ImaginaryTimeStatus, β::Real, ∇E2::Real, W::Real) -> st::ImaginaryTimeStatus

Update the imaginary-time status `st` with new values of `β`, `∇E2`, and `W`, and push the previous status into the history.
"""
function update!(st::ImaginaryTimeStatus, β::Real, ∇E2::Real, W::Real)
     @assert β ≥ 0 
     W < 0 && @warn "Estimated spectral width W = $(W) < 0"
     ∇E2 < 0 && @warn "Estimated ⟨∇E, ∇e⟩ = $(∇E2) < 0"
     push!(st.history, ImaginaryTimeStatus(st.β, st.∇E2, st.W))
     st.β = convert(Float64, β)
     st.∇E2 = convert(Float64, ∇E2)
     st.W = convert(Float64, W)
     return st
end

    

"""
     abstract type AbstractOptimizer end

Abstract type for optimizers in imaginary-time evolution algorithms. Each concrete optimizer should determine the next time step based on current status. A method like 

     (opt::T)(st::ImaginaryTimeStatus) -> τ::Float64

should be defined for any concrete subtype `T <: AbstractOptimizer`.
"""
abstract type AbstractOptimizer end


 
"""
     struct LinearOptimizer <: AbstractOptimizer
          τ::Float64
     end

The basic linear optimizer for imaginary-time evolution, which uses a fixed time step `τ`.
"""
struct LinearOptimizer <: AbstractOptimizer
     τ::Float64
     function LinearOptimizer(τ::Real)
          return new(convert(Float64, τ))
     end
end
(opt::LinearOptimizer)(::ImaginaryTimeStatus) = opt.τ

"""
     struct ExpLinearOptimizer <: AbstractOptimizer 
          τm::Float64
     end

The empirically commonly used optimizer in tanTRG, which uses an exponential cooling at the initial stage and then switches to linear cooling, controlled by a super parameter `τm` (maximum time step).
"""
struct ExpLinearOptimizer <: AbstractOptimizer 
     τm::Float64
     function ExpLinearOptimizer(τ::Real)
          return new(convert(Float64, τ))
     end
end
function (opt::ExpLinearOptimizer)(st::ImaginaryTimeStatus)::Float64 
     return st.β < opt.τm ? st.β : opt.τm 
end
   
""" 
     struct AdaptiveWidthOptimizer <: AbstractOptimizer
          a::Float64
          b::Float64 
     end

Adaptive optimizer based on estimated spectral width of Hamiltonian so that it is more robust against scaling energy unit. The time step is determined by 

     1/τ^2 = 1 / (a^2 β^2) + W^2 / b^2

with two super parameters `a` and `b`. The first term gives an exponential cooling warmup to avoid large time steps at the initial stage, while the second term adjusts the time step based on spectral width of the effective Hamiltonian.

# Constructor

     AdaptiveWidthOptimizer(a::Real = 1.0, b::Real = 10.0)
"""
struct AdaptiveWidthOptimizer <: AbstractOptimizer 
     a::Float64
     b::Float64

     function AdaptiveWidthOptimizer(a::Real = 1.0, b::Real = 10.0)
          @assert a > 0 
          @assert b > 0
          return new(convert(Float64, a), convert(Float64, b))
     end
end
function (opt::AdaptiveWidthOptimizer)(st::ImaginaryTimeStatus)::Float64 
     return 1 / sqrt(1 / (opt.a^2 * st.β^2) + st.W^2 / opt.b^2)
end

"""
     struct AdaptiveGradientOptimizer <: AbstractOptimizer
          a::Float64
          b::Float64 
          c::Float64
     end

Adaptive optimizer based on estimated gradient norm ⟨∇E, ∇e⟩, so that the step length decreases when the energy gradient is large. The time step is determined by 

     1/τ^2 = 1 / (a^2 β^2) + W^2 / b^2 + β^2 ⟨∇E, ∇e⟩^2 / c^2

with three super parameters `a`, `b`, and `c`. The first two terms are the same as in `AdaptiveWidthOptimizer`, while the last term is proportional to `(∂e/∂lnβ)^2`. When scaling energy unit by a factor `λ`, `τ` indeed scales by `1/λ`.

# Constructor

     AdaptiveGradientOptimizer(a::Real = 1.0, b::Real = 10.0, c::Real = 1.0)
"""
struct AdaptiveGradientOptimizer <: AbstractOptimizer 
     a::Float64
     b::Float64
     c::Float64
     function AdaptiveGradientOptimizer(a::Real = 1.0, b::Real = 10.0, c::Real = 1.0)
          @assert a > 0 
          @assert b > 0
          @assert c > 0
          return new(convert(Float64, a), convert(Float64, b), convert(Float64, c))
     end
end
function (opt::AdaptiveGradientOptimizer)(st::ImaginaryTimeStatus)::Float64 
     return 1 / sqrt(1 / (opt.a^2 * st.β^2) + st.W^2 / opt.b^2 + st.β^2 * st.∇E2^2 / opt.c^2)
end
     