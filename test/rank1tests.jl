module RankOneTest

using Test
using AMORS
using Random

bestscale(x, xref) = sum(x.*xref)/sum(x.*x)

contrast(A::AbstractArray) = (x = extrema(A); float(x[2]) - float(x[1]))
noisify(A::AbstractArray, snr) = noisify(Random.default_rng(), A, snr)
noisify(rng::AbstractRNG, A::AbstractArray, snr) =
    (contrast(A)/snr).*randn(rng, eltype(A), size(A)) + A
model(x::AbstractVector, y::AbstractVector) = x .* y'
function model(z::AbstractMatrix)
    x0 = sum(z; dims=2)[:,1];
    y0 = sum(z; dims=1)[1,:];
    alpha = bestscale(model(x0, y0), z);
    sx = sqrt(abs(alpha));
    sy = sign(alpha)*sx;
    return sx*x0, sy*y0
end

function runtest(; rng::AbstractRNG = MersenneTwister(314159),
                 maxiter::Integer = 200, xtol::Real = 1e-7, snr::Real = 10)
    # Ground-truth
    u = -3.2:0.1:5;
    xgt = sin.(u + 0.3.*u.*u);
    ygt = map(t -> abs(t) > 0.2 ? t : zero(t), rand(rng, Float64, 50) .- 0.1);
    zgt = xgt .* ygt';

    # Generate noisy data and initial solution. The higher the "SNR" the more difficult
    # the problem.
    z = noisify(rng, zgt, snr)
    x0, y0 = model(z)

    f = AMORS.RankOneProblem(z)

    io = stdout

    opts = (; io=io, observer=AMORS.observer, maxiter=maxiter, xtol=xtol)

    @testset "Solvers (autoscale = $(autoscale), first = Val(:$(xy)))" for xy in (:x, :y),
        autoscale in (true, false)

        println(io, "\n# Testing with `autoscale = $(autoscale)` and `first = Val(:$(xy))`")
        (info, x, y) = AMORS.solve(f, x0, y0; opts..., autoscale = autoscale, first = Val(xy));
        println(io, "# Final status: ", info.status)
        @test info.status ∈ (autoscale ? (:convergence,) : (:convergence, :too_many_iterations))
    end
    nothing
end

end # module RankOneTest
