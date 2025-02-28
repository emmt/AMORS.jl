using AMORS
using Test

@testset "AMORS.jl" begin

    @testset "Utilities" begin
        let T = Float32, dims = (2,3,4), x = rand(T, dims), y = similar(x), z = similar(x)
            # Test `scale!`
            @test AMORS.scale!(copyto!(y, x), -1.7) === y # in-place operation?
            @test AMORS.scale!(copyto!(y, x), 0) == zeros(T, dims)
            @test AMORS.scale!(fill!(y, NaN), 0) == zeros(T, dims)
            @test AMORS.scale!(copyto!(y, x), 1) == x
            @test AMORS.scale!(copyto!(y, x), -1) == -x
            @test AMORS.scale!(copyto!(y, x), 2) == 2*x

            # Test ` apply_scaling_factor!`
            @test AMORS.apply_scaling_factor!(one(T), copyto!(y, x), copyto!(z, x)) == 1
            @test y == x
            @test z == x
            @test AMORS.apply_scaling_factor!(T(2), copyto!(y, x), copyto!(z, x)) == 2
            @test y == T(2)*x
            @test z == x/T(2) # division by two is exact
            let alpha = T(0.217)
            @test AMORS.apply_scaling_factor!(alpha, copyto!(y, x), copyto!(z, x)) == alpha
                @test y ≈ alpha*x
                @test z ≈ x/alpha
            end

            # Test `has_converged`
            @. x += T(0.01) # make sure x > 0 everywhere
            @test AMORS.has_converged(x, x, 0) == true
            @test AMORS.has_converged(x, copyto!(y, x), 0) == true
            @. y = x*T(0.995) # so that ‖x - y‖ = 0.005*‖x‖
            @test AMORS.has_converged(x, y, 1e-2) == true
            @test AMORS.has_converged(x, y, 1e-3) == false
        end

        # Test `best_scaling_factor`
        let μ = 1e-2, Jx = 0.1, q = 2, ν = 0.7, Ky = 0.01, r = 3
            alpha = ((r*ν*Ky)/(q*μ*Jx))^inv(q + r)
            @test AMORS.best_scaling_factor(μ,Jx, q, ν,Ky, r) ≈ alpha
            @test AMORS.best_scaling_factor(μ*Jx, q, ν*Ky, r) ≈ alpha
            @test AMORS.best_scaling_factor(ν,Ky, r, μ,Jx, q) ≈ inv(alpha)
            @test AMORS.best_scaling_factor(ν*Ky, r, μ*Jx, q) ≈ inv(alpha)
            @test_throws DomainError AMORS.best_scaling_factor(0, Jx,  q,  ν, Ky,  r)
            @test_throws DomainError AMORS.best_scaling_factor(μ, Jx,  q,  0, Ky,  r)
            @test_throws DomainError AMORS.best_scaling_factor(μ, Jx, -q,  ν, Ky,  r)
            @test_throws DomainError AMORS.best_scaling_factor(μ, Jx,  q, -ν, Ky,  r)
            @test_throws DomainError AMORS.best_scaling_factor(μ, Jx,  q,  ν, Ky, -r)
        end
        let μ = 1.2e-3, Jx = 9.3e7, q = 1.4, ν = 3.2e1, Ky = 1.4e-3, r = 2
            alpha = ((r*ν*Ky)/(q*μ*Jx))^inv(q + r)
            @test AMORS.best_scaling_factor(μ,Jx, q, ν,Ky, r) ≈ alpha
            @test AMORS.best_scaling_factor(ν,Ky, r, μ,Jx, q) ≈ inv(alpha)
        end
    end

    @testset "Algorithm state" begin
        α = 0.91
        Gxy = 156.12
        μ = 10.7
        Jx = 37.1
        q = 1
        ν = 2.1
        Ky = 94.2
        r = 2
        autoscale = false
        iter = 237
        eval = 241
        status = :none
        A = @inferred AMORS.Info(α, Gxy, μ, Jx, q, ν, Ky, r, autoscale, iter, eval, status)
        @test A.α         == α
        @test A.Gxy       == Gxy
        @test A.μ         == μ
        @test A.Jx        == Jx
        @test A.q         == q
        @test A.ν         == ν
        @test A.Ky        == Ky
        @test A.r         == r
        @test A.autoscale == autoscale
        @test A.iter      == iter
        @test A.eval      == eval
        @test A.status    == status
        @test A.αbest     == AMORS.best_scaling_factor(μ, Jx, q, ν, Ky, r)
        @test A.η         == AMORS.effective_hyperparameter(μ, q, ν, r)
        @test A.Fxy       == AMORS.objective_function(Gxy, μ, Jx, q, ν, Ky, r, α)
    end

    include("rank1tests.jl")
    RankOneTest.runtest()
    nothing
end
