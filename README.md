# A Julia framework to apply the AMORS algorithm

[![Build Status](https://github.com/emmt/Amors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/Amors.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/Amors.jl?svg=true)](https://ci.appveyor.com/project/emmt/Amors-jl) [![Coverage](https://codecov.io/gh/emmt/Amors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/Amors.jl)

The `Amors` module provides a framework to apply the AMORS (*Alternated
Minimization using Optimal ReScaling*) algorithm for estimating the components `x`
and `y` of a *bilinear model* `x⋆y`.

The objective of AMORS is to minimize in `x ∈ X` and `y ∈ Y` an objective
function of the form:

``` julia
F(x,y) = G(x⋆y) + µ⋅J(x) + λ⋅K(y)
```

where `G` is a function of the *bilinear model* `x⋆y`, `J` and `K` are positive
homogeneous functions of the respective variables `x` and `y` while `µ > 0` and
`λ > 0` are so-called hyper-parameters. The notation `x⋆y` denotes a *bilinear
model* which has the following invariance property:

    (α⋅x)⋆(y/α) = x⋆y

for any factor `α > 0`.

The AMORS algorithm is described in:

1. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Exploiting the
   scaling indetermination of bi-linear models in inverse problems*", in 28th
   European Signal Processing Conference (EUSIPCO), pp. 2358–2362 (2021).
   [doi: 10.23919/Eusipco47968.2020.9287593]

2. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Unsupervised
   blind-deconvolution with optimal scaling applied to astronomical data*", in
   Adaptive Optics Systems VIII, International Society for Optics and Photonics
   (SPIE), Vol. 12185 (2022).
   [doi: 10.1117/12.2630245]
