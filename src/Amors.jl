"""

The `Amors` module provides a framework to apply the AMORS algorithm described
in:

1. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Exploiting the
   scaling indetermination of bi-linear models in inverse problems*", in 28th
   European Signal Processing Conference (EUSIPCO), pp. 2358–2362 (2021).
   [doi: 10.23919/Eusipco47968.2020.9287593]

2. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Unsupervised
   blind-deconvolution with optimal scaling applied to astronomical data*", in
   Adaptive Optics Systems VIII, International Society for Optics and Photonics
   (SPIE), Vol. 12185 (2022).
   [doi: 10.1117/12.2630245]

"""
module Amors

using Unitless

const default_atol = 0.3
const default_xtol = 1e-4
const default_maxiter = 1000

"""
    Amors.solve!(f, x, y, α = :auto) -> info, x, y

solves regularized *bilinear model* estimation by AMORS method. Object `f`
represents the objective function (see below). Arguments `x`, `y`, and `α` are
respectively the initial variables and factor of the problem. The result is a
3-tuple with the updated variables and `info` set so as to indicate the reason
of the algorithm termination: `info = :convergence` if algorithm has converged
within the given tolerances or `info = :too_many_iterations` if the algorithm
exceeded the maximum number of iterations.

The objective of AMORS is to minimize in `x ∈ X` and `y ∈ Y` an objective
function of the form:

    F(x,y) = G(x⋆y) + λ⋅J(x) + µ⋅K(y)

where `G` is a function of the *bilinear model* `x⋆y`, `J` and `K` are positive
homogeneous functions of the respective variables `x` and `y` while `λ > 0` and
`µ > 0` are so-called hyper-parameters. The notation `x⋆y` denotes a *bilinear
model* which has the following invariance property:

    (α⋅x)⋆(y/α) = x⋆y

for any factor `α > 0`.

The object `f` collects any data, workspaces, information, etc. needed to deal
with the objective function `F(x,y)`. This includes `X`, `Y`, `G`, `J`, `K`,
`λ`, and `µ`.

Note that thanks to the properties guaranteed by AMORS, the shapes of the
 components `x` and `y` depend on the tuning of only one of hyper-parameter
 (`λ` or `µ`, for instance).

The AMORS algorithm requires that methods `Amors.update!` and
`Amors.best_factor` be specialized for the types of `f`, `x`, and `y` so that:

    Amors.update!(Val(:x), f, x, y) ≈ min_{x ∈ X} F(x,y)
    Amors.update!(Val(:y), f, x, y) ≈ min_{y ∈ Y} F(x,y)

respectively update in-place the component `x` and `y` of the model and so
that:

    Amors.best_factor(f, x, y) -> α

yields the optimal value of the factor `α` such that `F(α⋅x,y/α)` is minimized
in `α`. As a helper, this latter method can be called as:

    Amors.best_factor(λ⋅J(x), q, µ⋅K(y), r)

to compute the best factor `α` given the current values of the terms `λ⋅J(x)`
and `µ⋅K(y)` and the homogeneous degrees `q` and `r` of the functions `J` and
`K` respectively. The initial factor `α` may be a value, or one of the symbolic
names `:auto` or `:const`. If a value is specified, it is used to scale the
initial variables and the best factor is used for every other iteration. If
`:auto` is specified `Amors.best_factor(f,x,y)` is always called to compute the
factor `α` (initially and at every iteration). If `:const` is specified, a
constant factor `α = 1` is always used (initially and at every iteration). This
latter possibility is not recommended, it is only useful for testing purposes.

Note that the `Amors.update!` method is always called with the current
(possibly pre-scaled) variables. This may be exploited to accelerate the
updating by not starting from scratch.

Arguments `x` and `y` are needed to define the variables. Initially, they must
be such that `J(x) > 0` and `K(y) > 0` if the factor `α` is automatically
computed.

The following keywords can be specified:

- `first` is a symbolic name which specifies which of `:x` or `:y` (the
  default) to update the first given the other.

- `atol` is a relative tolerance ($default_atol by default) to assert the
  convergence in the factor `α`.

- `xtol` is a relative tolerance ($default_xtol by default) to assert the
  convergence in the variables `x`.

- `ytol` is a relative tolerance (`xtol` by default) to assert the convergence
  in the variables `y`.

- `maxiter` is the maximum number of algorithm iterations ($default_maxiter by
  default).

- `conv` is a function used to check for convergence of the iterates
  (`Amors.check_convergence` by default).

"""
solve!(f, x, y, α::Symbol; kwds...) = solve!(f, x, y, Val(α); kwds...)

solve!(f, x, y, α::Val{:auto} = Val(:auto); kwds...) =
    solve!(f, x, y, best_factor(f, x, y); kwds...)

solve!(f, x, y, α::Val{:const}; kwds...) =
    solve!(f, x, y, 1.0; keep_α_fixed = true, kwds...)

# Catch errors:
solve!(f, x, y, ::Val{α}; kwds...) where {α} =
    throw(ArgumentError("invalid value `:$α` for parameter `α`"))

function solve!(f, x, y, α::Real;
                keep_α_fixed::Bool = false, # NOTE: change this option at your own risk!
                first::Symbol = :y,
                atol::Real = default_atol,
                xtol::Real = default_xtol,
                ytol::Real = xtol,
                maxiter::Integer = default_maxiter,
                conv::Function = check_convergence)
    keep_α_fixed && !isone(α) && throw(ArgumentError("initial value of `α` must be 1"))
    α > zero(α) || throw(ArgumentError("initial value of `α` must be strictly positive"))
    first ∈ (:x, :y) || throw(ArgumentError("value of keyword `first` must be `:x` or `:y`"))
    zero(atol) < atol < one(atol) || throw(ArgumentError("value of keyword `atol` must be in `(0,1)`"))
    zero(xtol) < xtol < one(xtol) || throw(ArgumentError("value of keyword `xtol` must be in `(0,1)`"))
    zero(ytol) < ytol < one(ytol) || throw(ArgumentError("value of keyword `ytol` must be in `(0,1)`"))
    maxiter ≥ 0 || throw(ArgumentError("value of keyword `maxiter` must be nonnegative"))
    x0 = similar(x)
    y0 = similar(y)
    iter = 0
    info = :work_in_progress
    α = Float64(α) # ensure type-stability
    while true # outer loop
        # Apply (initial or best) scaling factor.
        if α != one(α)
            scale!(x, α)
            scale!(y, one(α)/α)
        end

        # Check for convergence.
        if iter ≥ 1 && conv(x, x0, xtol) && conv(y, y0, ytol)
            info = :convergence
            break
        elseif iter ≥ maxiter
            info = :too_many_iterations
            break
        end

        # Save variables before their updating by this iteration.
        copyto!(y0, y)
        copyto!(x0, x)

        # Update first component given the orther, compute resulting best
        # scaling factor and apply it. If this is the first iteration, repeat
        # this step until the value of the best scaling factor converges.
        while true # inner loop
            if first === :y
                update!(Val(:y), f, x, y)
            else
                update!(Val(:x), f, x, y)
            end
            keep_α_fixed && break
            α = Float64(best_factor(f, x, y))
            if α != one(α)
                scale!(x, α)
                scale!(y, one(α)/α)
            end
            (iter > 0 || abs(α - one(α)) ≤ atol) && break
        end # inner loop

        # Update other component, compute resulting best scaling factor, and
        # proceed with next iterationq.
        if first === :y
            update!(Val(:x), f, x, y)
        else
            update!(Val(:y), f, x, y)
        end
        if keep_α_fixed == false
            α = Float64(best_factor(f, x, y))
        end
        iter += 1
    end # outer loop
    return info, x, y
end

"""
    Amors.check_convergence(x, xp, tol) -> bool

yields whether iterate `x` has converged. Argument `xp` is the previous value
of `x` and `tol ≥ 0` is a relative tolerance. The result is given by:

    ‖x - xp‖ ≤ tol⋅‖x‖

with `‖x‖` the Euclidean norm of `x`.

"""
function check_convergence(x::AbstractArray{T,N}, xp::AbstractArray{T,N},
                           tol::Real) where {T,N}
    a = b = abs2(zero(T))
    @inbounds @simd for i in eachindex(x, xp)
        a += abs2(x[i] - xp[i])
        b += abs2(x[i])
    end
    return sqrt(a) ≤ tol*sqrt(b)
end

"""
    Amors.scale!(x, α) -> x
    Amors.scale!(α, x) -> x

scale in-place the elements of the array `x` by the (unitless) factor `α`. If
`α == zero(α)` holds, `x` is zero-filled so its values may be initially
undefined.

"""
scale!(α::Number, x::AbstractArray{T}) where {T<:Number} = scale!(x, α)
function scale!(x::AbstractArray{T}, α::Number) where {T<:Number}
    if iszero(α)
        z = zero(T)
        @inbounds @simd for i in eachindex(x)
            x[i] = z
        end
    elseif !isone(α)
        a = convert_floating_point_type(T, α)
        @inbounds @simd for i in eachindex(x)
            x[i] *= a
        end
    end
    return x
end

"""
    Amors.best_factor(λ⋅J(x), q, µ⋅K(y), r) -> α

yields the best factor `α > 0` such that:

    λ⋅J(α⋅x) + µ⋅K(y/α) = α^q⋅λ⋅J(x) + µ⋅K(y)/α^r

is minimized in `α`. Arguments are the current values of the terms `λ⋅J(x)` and
`µ⋅K(y)` and the homogeneous degrees `q` and `r` of the fucntions `J` and `K`
respectively. This problem has the following closed-form solution:

    α = ((r⋅µ⋅K(y))/(q⋅λ⋅J(x)))^(1/(q + r))

which is returned by this function.

"""
best_factor(args::Vararg{Real,4}) = best_factor(map(Float64, args)...)
best_factor(λJx::Float64, q::Float64, µKy::Float64, r::Float64) =
    ((r*µKy)/(q*λJx))^(1.0/(q + r))

end
