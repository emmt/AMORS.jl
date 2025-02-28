const default_α = 1.0
const default_μ = 1.0
const default_ν = 1.0
const default_αtol = 0.3
const default_xtol = 1e-4
const default_maxiter = 1000
const default_Float = Float64

"""
    AMORS.solve(f, x0, y0) -> (status, x, y)

Apply AMORS strategy out-of-place, that is leaving the initial variables `x0` and `y0`
unchanged. See [`AMORS.solve!`](@ref) for a description of the method.

Methods `Base.similar` and `Base.copyto!` must be applicable to objects of same types as
`x0` and `y0`.

"""
solve(f, x0, y0; kwds...) = solve!(f, copy_variables(x0), copy_variables(y0); kwds...)

# This method only requires that `Base.copyto!` and `Base.similar` be applicable to the
# variables of the problem.
copy_variables(x) = copyto!(similar(x), x)

"""
    AMORS.solve!(f, x, y) -> info, x, y

Estimate the components of a *bilinear model* by the AMORS method. The argument `f`
represents the objective function (see below). On entry, arguments `x` and `y` are the
initial variables of the problem, they are overwritten by the solution (call
[`AMORS.solve`](@ref) for an out-of-place version of the algorithm). The result is a
3-tuple with the updated variables and `info` storing the final state of the algorithm.
For example, `info.status == :convergence` holds if algorithm has converged in the
variables `x` and `y` within the given tolerances or `info.status == :too_many_iterations`
holds if the algorithm exceeded the maximum number of iterations (see [`AMORS.Info`](@ref)
for more details).

The objective of AMORS is to minimize in `x ∈ 𝕏` and `y ∈ 𝕐` an objective function of the
form:

    F(x,y,μ,ν) = G(x⊗y) + μ*J(x) + ν*K(y)

where `G` is a function of the *bilinear model* `x⊗y`, `J` and `K` are nonnegative
homogeneous functions of the respective variables `x` and `y`, `μ > 0` and `ν > 0` are
hyper-parameters. The notation `x⊗y` denotes a *bilinear model* which has the following
invariance property:

    (α*x)⊗(y/α) = x⊗y

for any `x`, `y`, and scalar factor `α > 0`. An *homogeneous function*, say `J: 𝕏 → ℝ`, of
degree `q` is such that `J(α*x) = abs(α)^q*J(x)` for any `α ∈ ℝ` and for any `x ∈ 𝕏` with
`𝕏` the domain of `J`. It can be noted that the following property must hold `∀ α ∈ ℝ`: `x
∈ 𝕏` implies that `α*x ∈ 𝕏`. In other words, `𝕏` must be a cone.

The argument `f` is a callable object that collects any data, workspaces, parameters, etc.
needed to compute the objective function `F(x,y,μ,ν)`. The argument `f` is called as
`f(Val(task), args...)` where `task` is a symbolic name specifying the operation to be
performed:

    f(Val(:degJ))       -> deg(J)  # unless keyword `r` is specified
    f(Val(:degK))       -> deg(K)  # unless keyword `q` is specified
    f(Val(:Jx), x)      -> J(x)
    f(Val(:Ky), y)      -> K(y)
    f(Val(:x), x, y, μ) -> x⁺, G(x⁺⊗y), J(x⁺)
    f(Val(:y), x, y, ν) -> y⁺, G(x⊗y⁺), K(y⁺)

where `deg(J)` and `deg(K)` denote the respective homogeneous degrees of the functions `J`
and `K` and with:

    x⁺ ≈ argmin_{x ∈ 𝕏} G(x⊗y) + μ*J(x)
    y⁺ ≈ argmin_{y ∈ 𝕐} G(x⊗y) + ν*K(y)

The solution of `argmin_{x ∈ 𝕏} F(x, y)` and `argmin_{y ∈ 𝕐} F(x, y)` may not be exact and
may be computed in-place to save storage, that is `x` (resp. `y`) being overwritten by the
solution `x⁺` (resp. `y⁺`). For type stability of the algorithm,
`f(Val(:x),x,y,μ)[1]::typeof(x)` and `f(Val(:y),x,y,ν)[1]::typeof(y)` must hold.

Arguments `x` and `y` are needed to define the variables of the problem. Initially, they
must be such that `J(x) > 0` and `K(y) > 0` unless automatic best rescaling is disabled by
`autoscale=false` (which is not recommended).

The following keywords can be specified:

- `μ` is the multiplier of `J(x)`; `μ = $(default_μ)` by default.

- `ν` is the multiplier of `K(y)`; `ν = $(default_ν)` by default.

- `q` is the homogeneous degree of `J(x)`; `q = f(Val(:degJ))` by default.

- `r` is the homogeneous degree of `K(y)`; `r = f(Val(:degK))` by default.

- `α` is the initial scaling factor; `α = $(default_α)` by default. If `autoscale` is
  `false`, the value of `α` is unchanged for all iterations.

- `autoscale` specifies whether to automatically set the scaling factor `α`; `autoscale =
  true` by default. This keyword is provided for testing the efficiency of the AMORS
  algorithm, it is recommended to not disable autoscaling.

- `Float` is the floating-point type for scalar computations; `Float = $(default_Float)`
  by default.

- `first` is one of `Val(:x)` or `Val(:y)` (the default) to specify which component to
  update the first given the other.

- `αtol ≥ 0` is a relative tolerance (`αtol = $default_αtol` by default) to assert the
  convergence in the scaling factor `α` in the intial iteration of the algorithm. Use
  `αtol = Inf` to simply update `α` without checking for convergence. The value of `αtol`
  has no effects if `autoscale` is `false`.

- `xtol ≥ 0` is a relative tolerance (`xtol = $default_xtol` by default) to assert the
  convergence in the variables `x`.

- `ytol ≥ 0` is a relative tolerance (`ytol = xtol` by default) to assert the convergence
  in the variables `y`.

- `maxiter` is the maximum number of algorithm iterations (`$default_maxiter` by default).
  An iteration of the algorithm consists in updating one of the component of the bilinear
  model, `x` or `y`. Two iterations are therefore needed to completely update the model.

- `has_converged` is a function used to check for convergence of the iterates
  ([`AMORS.has_converged`](@ref) by default).

- `observer` is a user-defined function called after ever iteration as
  `observer(io,info,f,x,y)` with `io` the stream set by the corresponding keyword, `info`
  an instance of [`AMORS.Info`](@ref) with the current state of the algorithm, `f` the
  object defining the problem, and `x` and `y` the current estimates of the model
  components. This function may return a symbolic status, if this status is not
  `:searching`, then the algorithm will be terminated and the resulting `info` will be set
  with this status. The value returned by the observer is ignored if it is not a `Symbol`.

- `io` the stream for the observer, `stdout` by default.

"""
function solve!(f, x, y;
                first::Val = Val(:y),
                Float::Type{<:AbstractFloat} = default_Float,
                α::Real = default_α,
                autoscale::Bool = true,
                μ::Number = default_μ,
                ν::Number = default_ν,
                q::Real = f(Val(:degJ)),
                r::Real = f(Val(:degK)),
                αtol::Real = default_αtol,
                xtol::Real = default_xtol,
                ytol::Real = xtol,
                maxiter::Integer = default_maxiter,
                has_converged = AMORS.has_converged,
                observer = nothing,
                io::IO = stdout)
    # Check keyword values.
    first ∈ (Val(:x), Val(:y)) || throw(ArgumentError("keyword `first` must be `Val(:x)` or `Val(:y)`"))
    isconcretetype(Float) || throw(ArgumentError("keyword `Float` must be a concrete type, got `$Float`"))
    ispositive(α) || throw(ArgumentError("value of scaling factor `α` must be positive"))
    ispositive(μ) || throw(ArgumentError("value of multiplier `μ` must be positive"))
    ispositive(ν) || throw(ArgumentError("value of multiplier `ν` must be positive"))
    ispositive(q) || throw(ArgumentError("value of `q = deg(J)` must be positive"))
    ispositive(r) || throw(ArgumentError("value of `r = deg(K)` must be positive"))
    isnonnegative(αtol) || throw(ArgumentError("value of tolerance `αtol` must be nonnegative"))
    isnonnegative(xtol) || throw(ArgumentError("value of tolerance `xtol` must be nonnegative"))
    isnonnegative(ytol) || throw(ArgumentError("value of tolerance `ytol` must be nonnegative"))
    isnonnegative(maxiter) || throw(ArgumentError("maximum number of iterations `maxiter` must be nonnegative"))

    # Fix types of keyword values.
    α = as(Float, α)
    μ = convert_floating_point_type(Float, μ)
    ν = convert_floating_point_type(Float, ν)
    q = fix_degree(Float, q)
    r = fix_degree(Float, r)
    αtol = as(Float, αtol)
    xtol = as(Float, xtol)
    ytol = as(Float, ytol)
    maxiter = as(Int, maxiter)

    # First estimation is to discover the types returned by the user-defined function.
    if first === Val(:x)
        Ky = f(Val(:Ky), y)
        Gxy, Jx = solve_for_x!(f, x, y, μ*abs(α)^q)
        update_x = autoscale
    else
        Jx = f(Val(:Jx), x)
        Gxy, Ky = solve_for_y!(f, x, y, ν/abs(α)^r)
        update_x = !autoscale
    end
    isa(Gxy, Number) || error("`G(x⊗y)` is not a number, its type is `$(typeof(Gxy))`")
    isa(Jx, Number) || error("`J(x)` is not a number, its type is `$(typeof(Jx))`")
    isa(Ky, Number) || error("`K(y)` is not a number, its type is `$(typeof(Ky))`")
    Gxy = convert_floating_point_type(Float, Gxy)
    Jx = try
       as(typeof(one(Gxy)/one(μ)), Jx)
    catch
       error("types of `G(x⊗y)` and `μ*J(x)` are not compatible: `typeof(G(x⊗y)) = $(typeof(Gxy))`, `typeof(μ) = $(typeof(μ))`, and `typeof(J(x)) = $(typeof(Jx))`")
    end
    Ky = try
       as(typeof(one(Gxy)/one(ν)), Ky)
    catch
       error("types of `G(x⊗y)` and `ν*K(y)` are not compatible: `typeof(G(x⊗y)) = $(typeof(Gxy))`, `typeof(ν) = $(typeof(ν))`, and `typeof(K(y)) = $(typeof(Ky))`")
    end

    # Dispatch on types for other iterations.
    return solve!(f, x, y, update_x, α, autoscale, Gxy, μ, Jx, q, ν, Ky, r,
                  αtol, xtol, ytol, maxiter, has_converged, observer, io)
end

function solve!(f, x, y, update_x::Bool, α::Float, autoscale::Bool, Gxy::Number,
                μ::Number, Jx::Number, q::Union{Int,Float},
                ν::Number, Ky::Number, r::Union{Int,Float},
                αtol::Float, xtol::Float, ytol::Float,
                maxiter::Int, has_converged, observer, io::IO) where {Float<:AbstractFloat}
    # Each updating of `x` or `y` counts for an evaluation while each updating of `x` or
    # `y` with an accepted value of `α` counts for an iteration.
    eval = 1
    iter = (autoscale ? 0 : eval)
    xprev = similar(x)
    yprev = similar(y)
    x_has_converged = false
    y_has_converged = false
    status = :searching
    while true # Until convergence in `x` and `y`...
        # Some estimates of `x` and `y` are available. Call observer if any and check for
        # algorithm convergence or termination.
        if observer !== nothing
            rv = observer(io, Info(α, Gxy, μ, Jx, q, ν, Ky, r, autoscale, iter, eval, status), f, x, y)
            if rv isa Symbol && rv !== status
                # Observer has requested the algorithm to terminate.
                status = rv
                break
            end
        end
        if x_has_converged & y_has_converged
            # Convergence in the variables.
            status = :convergence
            break
        end
        if iter ≥ maxiter
            # Too many iterations
            status = :too_many_iterations
            break
        end

        # Memorize previous value of component to update.
        if update_x
            copyto!(xprev, x)
        else
            copyto!(yprev, y)
        end
        while true # Update `x` or `y` until convergence in `α`...
            if update_x
                Gxy, Jx = oftype((Gxy, Jx), solve_for_x!(f, x, y, μ*abs(α)^q))
            else
                Gxy, Ky = oftype((Gxy, Ky), solve_for_y!(f, x, y, ν/abs(α)^r))
            end
            eval += 1
            # Unless `α` must remain constant, compute optimal scaling factor `α`
            # iterating until convergence in `α` if this is the initial iteration of the
            # algorithm.
            autoscale || break
            αprev = α
            α = oftype(α, best_scaling_factor(μ*Jx, q, ν*Ky, r))
            (iter ≥ 1 || abs(α - αprev) ≤ αtol*abs(α)) && break
        end

        # A new iteration has been performed. Check for convergence in the updated
        # component.
        iter += 1
        if update_x
            x_has_converged = has_converged(x, xprev, xtol)
        else
            y_has_converged = has_converged(y, yprev, ytol)
        end

        # Toggle the component to update.
        update_x = !update_x
    end

    return Info(α, Gxy, μ, Jx, q, ν, Ky, r, autoscale, iter, eval, status), x, y
end

"""
    AMORS.has_converged(x, xp, tol) -> bool

yields whether the variables `x` has converged. Argument `xp` is the previous value of `x`
and `tol ≥ 0` is a relative tolerance.

In the default implementation provided by AMORS for `x` and `xp` being arrays, the
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
    AMORS.observer(io, info, f, x, y)

Observer that can be used in [`AMORS.solve`](@ref) or [`AMORS.solve!`](@ref).

"""
function observer(io::IO, info::Info, f, x, y)
    iter = info.iter
    eval = info.eval
    Fxy  = info.Fxy
    α    = info.α
    αbest = try
        best_scaling_factor(info)
    catch
        NaN
    end
    if iter == (info.autoscale ? 0 : 1)
        println(io, "#  ITER  EVAL          OBJFUN             ALPHA    BEST_ALPHA")
        println(io, "# -----------------------------------------------------------")
    end
    @printf io "  %5d %5d %23.15e %11.3e %11.3e\n" iter eval Fxy α αbest
    nothing
end

"""
    AMORS.scale!(x, α::Real) -> x
    AMORS.scale!(α::Real, x) -> x

Multiply in-place the entries of `x` by the scalar `α` and return `x`. Whatever the values
of the entries of `x`, nothing is done if `α = 1` and `x` is zero-filled if `α = 0`.

The AMORS package provides a default implementation of the method that is applicable to
any abstract array `x`. The method is expected to be extended for other types of argument
`x`.

See also `LinearAlgebra.lmul!(α::Number,x::AbstractArray)` and
`LinearAlgebra.rmul!(x::AbstractArray,α::Number)`.

"""
scale!(x::AbstractArray, α::Number) = scale!(α, x)
function scale!(α::Number, x::AbstractArray)
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
    info.αbest
    AMORS.best_scaling_factor(info::AMORS.Info)
    AMORS.best_scaling_factor(J(x), deg(J), K(y), deg(K))

yield the best scaling factor defined by:

    α⁺ = argmin_{α > 0} J(α*x) + K(y/α)

and which has a closed-form expression:

    α⁺ = ((deg(K)*K(y))/(deg(J)*J(x)))^(1/(deg(J) + deg(K)))

The arguments are the values of the homogeneous functions, `J(x)` and `K(y)`, and their
respective degrees `q = deg(J)` and `r = deg(K)` for the current estimates of the
variables `x` and `y` of a bilinear model. All these arguments may be provided by AMORS
algorithm state `info`.

"""
function best_scaling_factor(Jx::Number, degJ::Number, Ky::Number, degK::Number)
    Jx > zero(Jx) || throw(DomainError(Jx, "`J(x) > 0` must hold"))
    Ky > zero(Ky) || throw(DomainError(Ky, "`K(y) > 0` must hold"))
    degJ > zero(degJ) || throw(DomainError(degJ, "`deg(J) > 0` must hold"))
    degK > zero(degK) || throw(DomainError(degK, "`deg(K) > 0` must hold"))
    return ((degK*Ky)/(degJ*Jx))^inv(degJ + degK)
end

best_scaling_factor(A::Info) = best_scaling_factor(A.Jx, A.q, A.Ky, A.r)

"""
    info.Fxy
    AMORS.objective_function(info::AMORS.Info)
    AMORS.objective_function(G(x⊗y), μ, J(x), deg(J), ν, K(y), deg(K), α=1)

yield the value of the AMORS objective function:

    F(α*x, y/α, μ, ν) = F(x, y, μ*|α|^q, ν/|α|^r)
                      = G(x⊗y) + μ*J(x)*|α|^q + ν*K(y)/|α|^r

with `q = deg(J)` and `r = deg(K)` the homogeneous degrees of the functions `J(x)` and
`K(y)`. All required arguments may be provided by AMORS algorithm state `info`.

"""
function objective_function(Gxy::Number,
                            μ::Number, Jx::Number, q::Real,
                            ν::Number, Ky::Number, r::Real, α::Real = 1.0)
    return Gxy + μ*Jx*abs(α)^q + ν*Ky/abs(α)^r
end

objective_function(A::Info) = objective_function(A.Gxy, A.μ, A.Jx, A.q, A.ν, A.Ky, A.r, A.α)

"""
    info.η
    AMORS.effective_hyperparameter(info::AMORS.Info)
    AMORS.effective_hyperparameter(μ, q, ν, r)

yield the value of the effective hyper-parameter in AMORS algorithm:

    η = ((r/q)^qp + (q/r)^rp)*(μ^rp)*(ν^qp)

with:

    qp = q/(q + r)
    rp = r/(q + r)

and with `q = deg(J)` and `r = deg(K)` the homogeneous degrees of the functions `J(x)` and
`K(y)` and `μ > 0` and `ν > 0` their respective multipliers. All required arguments may be
provided by AMORS algorithm state `info`.

"""
function effective_hyperparameter(μ::Number, q::Real, ν::Number, r::Real)
    qp = q/(q + r)
    rp = r/(q + r)
    return ((r/q)^qp + (q/r)^rp)*(μ^rp)*(ν^qp)
end

effective_hyperparameter(A::Info) = effective_hyperparameter(A.μ, A.q, A.ν, A.r)

Base.propertynames(::Info) =
    (:α,
     :αbest,
     :η,
     :Fxy,
     :Gxy,
     :μ,
     :Jx,
     :q,
     :ν,
     :Ky,
     :r,
     :autoscale,
     :iter,
     :eval,
     :status,)

Base.getproperty(A::Info, key::Symbol) =
    key === :α         ? getfield(A, :α        )     :
    key === :Gxy       ? getfield(A, :Gxy      )     :
    key === :μ         ? getfield(A, :μ        )     :
    key === :Jx        ? getfield(A, :Jx       )     :
    key === :q         ? getfield(A, :q        )     :
    key === :ν         ? getfield(A, :ν        )     :
    key === :Ky        ? getfield(A, :Ky       )     :
    key === :r         ? getfield(A, :r        )     :
    key === :autoscale ? getfield(A, :autoscale)     :
    key === :iter      ? getfield(A, :iter     )     :
    key === :eval      ? getfield(A, :eval     )     :
    key === :status    ? getfield(A, :status   )     :
    key === :αbest     ? best_scaling_factor(A)      :
    key === :η         ? effective_hyperparameter(A) :
    key === :Fxy       ? objective_function(A)       :
    throw(KeyError(key))

# Predicates.
ispositive(x::Number) = x > zero(x)
isnonnegative(x::Number) = x ≥ zero(x)

# Solve for component `x` making sure operation is done in-place.
function solve_for_x!(f, x, y, μ::Number)
    xnew, Gxy, Jx = f(Val(:x), x, y, μ)
    xnew === x || copyto!(x, xnew)
    return Gxy, Jx
end

# Solve for component `y` making sure operation is done in-place.
function solve_for_y!(f, x, y, ν::Number)
    ynew, Gxy, Ky = f(Val(:y), x, y, ν)
    ynew === y || copyto!(y, ynew)
    return Gxy, Ky
end

# Homogeneous degree must be an integer or a floating-point.
fix_degree(::Type{Float}, x::Integer) where {Float<:AbstractFloat} = as(Int, x)
fix_degree(::Type{Float}, x::Real   ) where {Float<:AbstractFloat} = as(Float, x)
