"""
    info = AMORS.Info(α, Gxy, μ, Jx, q, ν, Ky, r, iter, eval, status)

builds a structured object storing all informations about `AMORS` algorithm state. This
kind of object is returned by [`AMORS.solve`](@ref) and [`AMORS.solve!`](@ref)

The properties of the object have the same names as the arguments of the constructor:

    info.α      # scaling factor
    info.Gxy    # value of G(x⊗y)
    info.μ      # value of hyper-parameter for J(x)
    info.J      # value of J(x)
    info.q      # homogeneous degree of J(x)
    info.ν      # value of hyper-parameter for K(y)
    info.Ky     # value of K(y)
    info.r      # homogeneous degree of K(y)
    info.iter   # number of iterations
    info.eval   # number of "evaluations"
    info.status # current status of the algorithm

Call `AMORS.best_scaling_factor(info)` to comnpute the best possible value of `α` which
may not be `info.α` if autoscaling has been disabled in [`AMORS.solve`](@ref) or
[`AMORS.solve!`](@ref).

Call `AMORS.objective_function(info)` to compute the value of the objective function that
is to be minimized by the `AMORS` algorithm in `x`, `y`, and possibly `α`:

    F(α*x, y/α, μ, ν) = F(x, y, μ*|α|^q, ν/|α|^r)
                      = G(x⊗y) + μ*J(x)*|α|^q + ν*K(y)/|α|^r

"""
struct Info{TA<:AbstractFloat,TG<:Number,
            TM<:Number,TJ<:Number,TQ<:Union{Int,TA},
            TN<:Number,TK<:Number,TR<:Union{Int,TA}}
    α::TA          # scaling factor
    Gxy::TG        # value of G(x⊗y)
    μ::TM          # value of hyper-parameter for J(x)
    Jx::TJ         # value of J(x)
    q::TQ          # homogeneous degree of J(x)
    ν::TN          # value of hyper-parameter for K(y)
    Ky::TK         # value of K(y)
    r::TR          # homogeneous degree of K(y)
    iter::Int      # number of iterations
    eval::Int      # number of "evaluations"
    status::Symbol # current status of the algorithm
    function Info(α::TA, Gxy::TG,
                  μ::TM, Jx::TJ, q::TQ,
                  ν::TN, Ky::TK, r::TR,
                  iter::Int, eval::Int,
                  status::Symbol) where {TA<:AbstractFloat,TG<:Number,
                                         TM<:Number,TJ<:Number,TQ<:Union{Int,TA},
                                         TN<:Number,TK<:Number,TR<:Union{Int,TA}}
        return new{TA,TG,TM,TJ,TQ,TN,TK,TR}(α, Gxy, μ, Jx, q, ν, Ky, r, iter, eval, status)
    end
end
