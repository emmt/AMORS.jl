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
        let Jx = 0.1, degJ = 2, Ky = 0.01, degK = 3, alpha = ((degK*Ky)/(degJ*Jx))^inv(degJ + degK)
            @test AMORS.best_scaling_factor(Jx, degJ, Ky, degK) ≈ alpha
            @test AMORS.best_scaling_factor(Ky, degK, Jx, degJ) ≈ inv(alpha)
            @test_throws DomainError AMORS.best_scaling_factor(0.0, degJ, Ky, degK)
            @test_throws DomainError AMORS.best_scaling_factor(Jx, -0.1, Ky, degK)
            @test_throws DomainError AMORS.best_scaling_factor(Jx, degJ, -1e2, degK)
            @test_throws DomainError AMORS.best_scaling_factor(Jx, degJ, Ky, -1.0)
        end
        let Jx = 9.3e7, degJ = 1, Ky = 1.4e-3, degK = 2, alpha = ((degK*Ky)/(degJ*Jx))^inv(degJ + degK)
            @test AMORS.best_scaling_factor(Jx, degJ, Ky, degK) ≈ alpha
            @test AMORS.best_scaling_factor(Ky, degK, Jx, degJ) ≈ inv(alpha)
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
        @test A.αbest     == AMORS.best_scaling_factor(Jx, q, Ky, r)
        @test A.η         == AMORS.effective_hyperparameter(μ, q, ν, r)
        @test A.Fxy       == AMORS.objective_function(Gxy, μ, Jx, q, ν, Ky, r, α)
    end

    include("rank1tests.jl")
    RankOneTest.runtest()
    nothing
end
