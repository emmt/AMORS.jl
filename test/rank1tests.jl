module RankOneTest

using AMORS
using PythonPlot
const plt = PythonPlot

bestscale(x, xref) = sum(x.*xref)/sum(x.*x)

contrast(A::AbstractArray) = (x = extrema(A); float(x[2]) - float(x[1]))
noisify(A::AbstractArray, snr) =
    (contrast(A)/snr).*randn(eltype(A), size(A)) + A
model(x::AbstractVector, y::AbstractVector) = x .* y'
function model(z::AbstractMatrix)
    x0 = sum(z; dims=2)[:,1];
    y0 = sum(z; dims=1)[1,:];
    alpha = bestscale(model(x0, y0), z);
    sx = sqrt(abs(alpha));
    sy = sign(alpha)*sx;
    return sx*x0, sy*y0
end

# Ground-truth
u = -3.2:0.1:5;
xgt = sin.(u + 0.3.*u.*u);
ygt = map(t -> abs(t) > 0.2 ? t : zero(t), rand(Float64, 50) .- 0.1);
zgt = xgt .* ygt';

# Generate noisy data and initial solution. The higher the "SNR" the more difficult
# the problem.
z = noisify(zgt, 8)
x0, y0 = model(z)

f = AMORS.RankOneProblem(z)

println("\n# Testing with `autoscale=true`, `first=Val(:x)`")
(info, x, y) = AMORS.solve(RankOneTest.f, RankOneTest.x0, RankOneTest.y0;
                           observer=AMORS.observer, maxiter=500, autoscale=true,
                           xtol=1e-7, first=Val(:x));

println("\n# Testing with `autoscale=false`, `first=Val(:x)`")
(info, x, y) = AMORS.solve(RankOneTest.f, RankOneTest.x0, RankOneTest.y0;
                           observer=AMORS.observer, maxiter=500, autoscale=false,
                           xtol=1e-7, first=Val(:x));

println("\n# Testing with `autoscale=true`, `first=Val(:y)`")
(info, x, y) = AMORS.solve(RankOneTest.f, RankOneTest.x0, RankOneTest.y0;
                           observer=AMORS.observer, maxiter=500, autoscale=true,
                           xtol=1e-7, first=Val(:y));

println("\n# Testing with `autoscale=false`, `first=Val(:y)`")
(info, x, y) = AMORS.solve(RankOneTest.f, RankOneTest.x0, RankOneTest.y0;
                           observer=AMORS.observer, maxiter=500, autoscale=false,
                           xtol=1e-7, first=Val(:y));

end # module RankOneTest
