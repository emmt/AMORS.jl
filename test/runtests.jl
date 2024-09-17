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

    include("rank1tests.jl")

end
