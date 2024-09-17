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
mutable struct RankOneProblem{T<:AbstractFloat}
    z::Matrix{T}
    function RankOneProblem(z::AbstractMatrix{<:Real})
        T = float(eltype(z))
        return new{T}(z)
    end
end

# Compute `G(x⊗y)`.
function (f::RankOneProblem)(::Val{:Gxy}, x::AbstractVector, y::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)

    Gxy = abs2(zero(eltype(z)) - zero(eltype(x))*zero(eltype(y)))
    @inbounds for j in J
        s = zero(Gxy)
        for i in I
            s += abs2(z[i,j] - x[i]*y[j])
        end
        Gxy += s
    end
    return Gxy
end

# Yield homogeneous degree of `J(x)`.
(f::RankOneProblem)(::Val{:degJ}) = 2

# Compute `J(x)`.
function (f::RankOneProblem)(::Val{:Jx}, x::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)

    # Compute regularization on x.
    Jx = zero(promote_type(Float64, float(eltype(x))))
    @inbounds for i in first(I):last(I)-1
        Jx += oftype(Jx, abs2(x[i+1] - x[i]))
    end
    return Jx
end

# Yield homogeneous degree of `K(y)`.
(f::RankOneProblem)(::Val{:degK}) = 2

# Compute `K(y)`.
function (f::RankOneProblem)(::Val{:Ky}, y::AbstractVector)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(y) == (J,)

    # Compute regularization on y.
    Ky = zero(promote_type(Float64, float(eltype(y))))
    @inbounds for j in J
        Ky += oftype(Ky, abs2(y[j]))
    end
    return Ky
end

# Fit `x` given `y`.
function (f::RankOneProblem)(::Val{:x}, x::AbstractVector, y::AbstractVector, μ::Number)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)
    @assert first(I) == 1
    @assert first(J) == 1
    m, n = length(I), length(J)

    # Compute the coefficients of the normal equations for x.
    T = float(promote_type(eltype(z), typeof(zero(eltype(x))*zero(eltype(y)))))
    d = Array{T}(undef, m) # diagonal entries
    e = Array{T}(undef, m - 1) # sub-diagonal entries
    q = T(mapreduce(abs2, +, y))
    r = as(T, μ)
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

    # Solve the normal equations for `x`.
    @static if VERSION < v"1.4"
        copyto!(x, A\b)
    else
        ldiv!(x, A, b)
    end

    # Return updated variable and costs.
    Gxy = f(Val(:Gxy), x, y)
    Jx = f(Val(:Jx), x)
    return x, Gxy, Jx
end

function (f::RankOneProblem)(::Val{:y}, x::AbstractVector, y::AbstractVector, ν::Number)
    # Extract sizes and check.
    z = f.z
    (I, J) = axes(z)
    @assert axes(x) == (I,)
    @assert axes(y) == (J,)
    @assert first(I) == 1
    @assert first(J) == 1
    m, n = length(I), length(J)

    # Compute the coefficients of the normal equations for y.
    T = float(promote_type(eltype(z), typeof(zero(eltype(x))*zero(eltype(y)))))
    a = T(mapreduce(abs2, +, x)) + T(ν)
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

    # Return updated variable and costs.
    Gxy = f(Val(:Gxy), x, y)
    Ky = f(Val(:Ky), y)
    return y, Gxy, Ky
end

end # module RankOne
