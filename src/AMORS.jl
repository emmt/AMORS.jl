"""

`AMORS` provides an implementation of the *Alternated Minimization using Optimal
ReScaling* method described in:

1. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Exploiting the scaling
   indetermination of bi-linear models in inverse problems*", in 28th European Signal
   Processing Conference (EUSIPCO), pp. 2358–2362 (2021)
   [DOI](https://doi.org/10.23919/Eusipco47968.2020.9287593).

2. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Unsupervised
   blind-deconvolution with optimal scaling applied to astronomical data*", in Adaptive
   Optics Systems VIII, International Society for Optics and Photonics (SPIE), Vol. 12185
   (2022) [DOI](https://doi.org/10.1117/12.2630245).

"""
module AMORS

using TypeUtils

const default_atol = 0.3
const default_xtol = 1e-4
const default_maxiter = 1000

"""
    AMORS.solve(f, x0, y0) -> (status, x, y)

Apply AMORS strategy out-of-place, that is leaving the intial variables `x0` and `y0`
unchanged. See [`AMORS.solve!`](@ref) for a description of the method.

Methods `Base.similar` and `Base.copyto!` must be applicable to objects of same types as
`x0` and `y0`.

"""
solve(f, x0, y0; kwds...) = solve!(f, copy_variables(x0), copy_variables(y0); kwds...)

# This method only requires that `Base.copyto!` and `Base.similar` be applicable to the
# variables of the problem.
copy_variables(x) = copyto!(similar(x), x)

"""
    AMORS.solve!(f, x, y) -> status, x, y

Estimnate the components of a *bilinear model* by the AMORS method. The argument `f`
represents the objective function (see below). On entry, arguments `x` and `y` are the
initial variables of the problem, they are overwritten by the solution. The result is a
3-tuple with the updated variables and `status` indicating the reason of the algorithm
termination: `status = :convergence` if algorithm has converged in the variables `x` and
`y` within the given tolerances or `status = :too_many_iterations` if the algorithm
exceeded the maximum number of iterations.

The objective of AMORS is to minimize in `x ∈ 𝕏` and `y ∈ 𝕐` an objective function of the
form:

    F(x,y) = G(x⊗y) + J(x) + K(y)

where `G` is a function of the *bilinear model* `x⊗y`, `J` and `K` are positive
homogeneous functions of the respective variables `x` and `y`. The notation `x⊗y` denotes
a *bilinear model* which has the following invariance property:

    (α*x)⊗(y/α) = x⊗y

for any scalar factor `α > 0`. An *homogeneous function*, say `J: 𝕏 → ℝ`, of degree `q` is
such that `J(α*x) = abs(α)^q*J(x)` for any `α ∈ ℝ` and for any `x ∈ 𝕏` with `𝕏` the domain
of `J`. It can be noted that the following property must hold `∀ α ∈ ℝ`: `x ∈ 𝕏` implies
that `α*x ∈ 𝕏`. In other words, `𝕏` must be a cone.

The argument `f` collects any data, workspaces, parameters, etc. needed to deal with the
objective function `F(x,y)`. This includes `𝕏`, `𝕐`, `G`, `J`, and `K`. The argument `f`
must be a callable object which is called as:

    f(task, x, y)

where `task` is `Val(:x)` or `Val(:y)` to update this component:

    f(Val(:x), x, y) -> argmin_{x ∈ 𝕏} F(x, y) = argmin_{x ∈ 𝕏} G(x⊗y) + J(x)
    f(Val(:y), x, y) -> argmin_{y ∈ 𝕐} F(x, y) = argmin_{y ∈ 𝕐} G(x⊗y) + K(y)

while `task` is `Val(:alpha)` to yield the optimal scaling `α > 0`:

    f(Val(:alpha), x, y) -> argmin_{α > 0} F(α*x, y/α) = argmin_{α > 0} J(α*x) + K(y/α)

The solution of `argmin_{x ∈ 𝕏} F(x, y)` and `argmin_{y ∈ 𝕐} F(x, y)` may not be exact and
may be computed in-place to save storage, that is `x` (resp. `y`) being overwritten by the
solution. For type stability of the algorithm, `f(Val(:x),x,y)::typeof(x)` and
`f(Val(:y),x,y)::typeof(y)` must hold.

The solution of `argmin_{α > 0} F(α*x, y/α)` has a closed-form expression:

    argmin_{α > 0} F(α*x, y/α) = ((deg(K)*K(y))/(deg(J)*J(x)))^(inv(deg(J) + deg(K)))

where `deg(J)` denotes the degree of the homogeneous function `J`. This solution can be
computed by calling [`AMORS.best_scaling_factor`](@ref).

Arguments `x` and `y` are needed to define the variables. Initially, they must be such
that `J(x) > 0` and `K(y) > 0` unless automatic best rescaling is disabled by
`do_not_scale=true` (which is not recommended).

The following keywords can be specified:

- `first` is one of `Val(:x)` or `Val(:y)` (the default) to specify which component to
  update the first given the other.

- `atol` is a relative tolerance ($default_atol by default) to assert the convergence in
  the factor `α`.

- `xtol` is a relative tolerance ($default_xtol by default) to assert the convergence in
  the variables `x`.

- `ytol` is a relative tolerance (`xtol` by default) to assert the convergence in the
  variables `y`.

- `maxiter` is the maximum number of algorithm iterations (`$default_maxiter` by default).

- `has_converged` is a function used to check for convergence of the iterates
  ([`AMORS.has_converged`](@ref) by default).

- `do_not_scale` may be set to `true` (default is `false`) to not scale the components `x`
  and `y` of the problem. This keyword is provided for testing the efficiency of the AMORS
  strategy, it is recommended to not use it.

"""
function solve!(f, x, y;
                first::Val = Val(:y),
                atol::Real = default_atol,
                xtol::Real = default_xtol,
                ytol::Real = xtol,
                maxiter::Integer = default_maxiter,
                has_converged = AMORS.has_converged,
                observer = nothing,
                do_not_scale::Bool = false)
    # Check keywords.
    first ∈ (Val(:x), Val(:y)) || throw(ArgumentError("bad value for keyword `first`, must be `Val(:x)` or `Val(:y)`"))
    zero(atol) < atol < one(atol) || throw(ArgumentError("value of keyword `atol` must be in `(0,1)`"))
    zero(xtol) < xtol < one(xtol) || throw(ArgumentError("value of keyword `xtol` must be in `(0,1)`"))
    zero(ytol) < ytol < one(ytol) || throw(ArgumentError("value of keyword `ytol` must be in `(0,1)`"))
    maxiter ≥ 0 || throw(ArgumentError("value of keyword `maxiter` must be nonnegative"))

    # Initialize algorithm.
    iter = 0
    xp = similar(x)
    yp = similar(y)
    status = :searching
    while true
        # Inspect iterate if requested.
        observer === nothing || observer(iter, f, x, y)

        # Check for convergence.
        if iter > 1 && has_converged(x, xp, xtol) && has_converged(y, yp, ytol)
            status = :convergence # convergence in the variables
            break
        elseif iter ≥ maxiter
            status = :too_many_iterations # too many iterations
            break
        end

        # Memorize the components of the problem before updating.
        copyto!(xp, x)
        copyto!(yp, y)

        # Update first component and re-scale. If this is the initial iteration, repeat
        # until convergence in the scaling factor.
        while true
            if first === Val(:x)
                x = f(Val(:x), x, y)::typeof(x)
            else
                y = f(Val(:y), x, y)::typeof(y)
            end
            do_not_scale && break
            α = apply_scaling_factor!(f(Val(:alpha), x, y), x, y)
            if iter ≥ 1 || abs(α - one(α)) ≤ atol
                break
            end
        end

        # Update second component and re-scale.
        if first === Val(:x)
            y = f(Val(:y), x, y)::typeof(y)
        else
            x = f(Val(:x), x, y)::typeof(x)
        end
        do_not_scale || apply_scaling_factor!(f(Val(:alpha), x, y), x, y)

        # Iteration completed.
        iter += 1
    end
    return status, x, y
end

"""
    AMORS.has_converged(x, xp, tol) -> bool

yields whether the variables `x` has converged. Argument `xp` is the previous value of `x`
and `tol ≥ 0` is a relative tolerance.

In the default implementation provided by `AMORS` for `x` and `xp` being arrays, the
result is given by:

    ‖x - xp‖ ≤ tol⋅‖x‖

with `‖x‖` the Euclidean norm of `x`.

The method is expected to be extended for non-array types of `x` and `xp`. Another
possibility is to specify the keyword `has_converged` in the call to [`AMORS.solve`](@ref)
or [`AMORS.solve!`](@ref).

"""
function has_converged(x::AbstractArray, xp::AbstractArray, tol::Real)
    axes(x) == axes(xp) || throw(DimensionMismatch("arrays must have the same axes"))
    s = abs2(zero(eltype(x)))
    d = abs2(zero(eltype(x)) - zero(eltype(xp)))
    @inbounds @simd for i in eachindex(x, xp)
        s += abs2(x[i])
        d += abs2(x[i] - xp[i])
    end
    return sqrt(d) ≤ tol*sqrt(s)
end

"""
    AMORS.scale!(x, α::Real) -> x
    AMORS.scale!(α::Real, x) -> x

Multiply in-place the entries of `x` by the scalar `α` and return `x`. Whatever the values
of the entries of `x`, nothing is done if `α = 1` and `x` is zero-filled if `α = 0`.

The `AMORS` package provides a default implementation of the method that is applicable to
any abstract array `x`. The method is expected to be extended for other types of argument
`x`.

See also `LinearAlgebra.lmul!(α::Number,x::AbstractArray)` and
`LinearAlgebra.rmul!(x::AbstractArray,α::Number)`.

"""
scale!(x::AbstractArray, α::Number, ) = scale!(x, α)
function scale!(α::Real, x::AbstractArray)
    if iszero(α)
        fill!(x, zero(eltype(x)))
    else !isone(α)
        α = convert_floating_point_type(eltype(x), α)
        @inbounds @simd for i in eachindex(x)
            x[i] *= α
        end
    end
    return x
end

"""
    AMORS.apply_scaling_factor!(α::Real, x, y) -> α

Multiply in-place the entries of `x` by the scalar `α` and the entries of `y` by `inv(α)`.
Return `α`. See [`AMORS.scale!`](@ref).

"""
function apply_scaling_factor!(α::Real, x, y)
    if !isone(α)
        scale!(α, x)
        scale!(inv(α), y)
    end
    return α
end

"""
    AMORS.best_scaling_factor(J(x), deg(J), K(y), deg(K)) -> α⁺

yields the best scaling factor defined by:

    α⁺ = argmin_{α > 0} J(α*x) + K(y/α)

and which has a closed-form expression:

    α⁺ = ((deg(K)*K(y))/(deg(J)*J(x)))^(1/(deg(J) + deg(K)))

The arguments are the values of the homogeneous objective functions, `J(x)` and `K(y)`,
and their respective degrees `deg(J)` and `deg(K)` for the current estimates of the
variables `x` and `y` of a bilinear model.

"""
best_scaling_factor(Jx::Real, degJ::Real, Ky::Real, degK::Real) =
    ((degK*Ky)/(degJ*Jx))^(inv(degJ + degK))

end
