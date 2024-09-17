module RankOneTest

using AMORS
using PythonPlot
const plt = PythonPlot

bestscale(x, xref) = sum(x.*xref)/sum(x.*x)
observer(info, f, x, y) = println(info.iter, " / ", info.eval, " -> ", AMORS.objective_function(info))
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

#(info, x, y) = AMORS.solve(f,x0,y0; observer, maxiter=50,autoscale=true,xtol=1e-7,first=Val(:x));

end # module RankOneTest
