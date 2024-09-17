module RankOne

import ..AMORS
using TypeUtils
using LinearAlgebra

"""
    f = AMORS.RankOneProblem(...)

Build an object representing a problem with a bilinear model given by a rank-1 matrix:

    (x⊗y)[i,j] = x[i]*y[j]
    G(x⊗y) = sum_{i,j} ((x⊗y)[i,j] - z[i,j])^2
    J(x) = sum_{i} (x[i+1] - x[i])^2
    K(y) = sum_{j} y[j]^2

where `z` is the input data.

"""
mutable struct RankOneProblem{T}
    z::Matrix{T}
    μ::Float64
    ν::Float64
    Fxy::Float64
    Gxy::Float64
    Jx::Float64
    Ky::Float64
    function RankOneProblem(z::AbstractMatrix, x::AbstractVector, y::AbstractVector; μ::Number, ν::Number)
        T = floating_point_type(eltype(z), eltype(x), eltype(y))
        f = new{T}(z, μ, ν, NaN, NaN, NaN, NaN)
        f(Val(:obj), x, y)
        return f
    end
end

# Compute objective function.
function (f::RankOneProblem)(::Val{:obj}, x::AbstractVector, y::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)

    # Compute data fidelity term.
    Gxy = zero(Float64)
    @inbounds for j in J
        s = zero(Float64)
        for i in I
            s += Float64(abs2(x[i]*y[j] - z[i,j]))
        end
        Gxy += s
    end
    f.Gxy = Gxy

    # Compute regularization on x.
    f.Jx = f(Val(:Jx), x, y)

    # Compute regularization on y.
    f.Ky = f(Val(:Ky), x, y)

    # Compute total objective function.
    f.Fxy = f.Gxy + f.Jx + f.Ky
    return f.Fxy
end

# Compute `μ*J(x)`.
function (f::RankOneProblem)(::Val{:Jx}, x::AbstractVector, y::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)

    # Compute regularization on x.
    Jx = zero(Float64)
    @inbounds for i in first(I):last(I)-1
        Jx += Float64(abs2(x[i+1] - x[i]))
    end
    return f.μ*Jx
end

# Compute `ν*K(y)`.
function (f::RankOneProblem)(::Val{:Ky}, x::AbstractVector, y::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)

    # Compute regularization on y.
    Ky = zero(Float64)
    @inbounds for j in J
        Ky += Float64(abs2(y[j]))
    end
    return f.ν*Ky
end

function (f::RankOneProblem{T})(::Val{:x}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)
    @assert first(I) == 1
    @assert first(J) == 1
    m, n = length(I), length(J)

    # Compute the coefficients of the normal equations for x.
    d = Array{T}(undef, m) # diagonal entries
    e = Array{T}(undef, m - 1) # sub-diagonal entries
    q = T(mapreduce(abs2, +, y))
    #println(q)
    r = T(f.μ)
    fill!(e, -r) # FIXME use UniformVector
    if m == 1
        d[1] = q
    elseif m ≥ 2
        d[1] = q + r
        @inbounds for i in 2:m-1
            d[i] = q + 2r
        end
        d[m] = q + r
    end
    A = SymTridiagonal(d, e)
    b = zeros(T, m)
    @inbounds for j in 1:n
        for i in 1:m
             b[i] += z[i,j]*y[j]
        end
    end

    # Solve the normal equations for x and update f.
    ldiv!(x, A, b)
    f(Val(:obj), x, y)
    return x
end

function (f::RankOneProblem{T})(::Val{:y}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)
    @assert first(I) == 1
    @assert first(J) == 1
    m, n = length(I), length(J)

    # Compute the coefficients of the normal equations for y.
    a = T(mapreduce(abs2, +, x) + f.ν)
    b = Array{T}(undef, n)
    @inbounds for j in 1:n
        s = zero(T)
        for i in 1:m
            s += z[i,j]*x[i]
        end
        b[j] = s
    end

    # Solve the normal equations for y and update f.
    @inbounds for j in 1:n
        y[j] = b[j]/a
    end
    f(Val(:obj), x, y)
    return y
end


function (f::RankOneProblem{T})(::Val{:alpha}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    Jx = f(Val(:Jx), x, y)
    Ky = f(Val(:Ky), x, y)
    return AMORS.best_scaling_factor(Jx, 2, Ky, 2)
end

end # module RankOne
