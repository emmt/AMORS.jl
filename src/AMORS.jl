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

include("compat.jl")

@public(Info, solve, solve!, has_converged, observer, scale!, apply_scaling_factor!,
        best_scaling_factor, objective_function, effective_hyperparameter)

using Printf
using TypeUtils

include("types.jl")
include("solver.jl")
include("rank1.jl")
import .RankOne: RankOneProblem

end # module AMORS
