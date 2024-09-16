# A Julia framework to apply the AMORS algorithm

[![Build Status](https://github.com/emmt/AMORS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/AMORS.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/AMORS.jl?svg=true)](https://ci.appveyor.com/project/emmt/AMORS-jl) [![Coverage](https://codecov.io/gh/emmt/AMORS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/AMORS.jl)

This Julia package provides an implementation of the `AMORS` algorithm (for *Alternated
Minimization using Optimal ReScaling*) for estimating the components `x` and `y` of a
*bilinear model* `x⊗y`.

The objective of `AMORS` is to minimize in `x ∈ 𝕏` and `y ∈ 𝕐` an objective function of
the form:

``` julia
F(x,y) = G(x⊗y) + J(x) + K(y)
```

where `G` is a function of the *bilinear model* `x⊗y` and where `J` and `K` are positive
homogeneous functions of the respective variables `x`. The notation `x⊗y` denotes a
*bilinear model* which has the following invariance property:

``` julia
(α*x)⊗(y/α) = x⊗y
```

for any scalar factor `α > 0`.

An *homogeneous function*, say `J: 𝕏 → ℝ`, of degree `q` is such that `J(α*x) =
abs(α)^q*J(x)` for any `α ∈ ℝ` and for any `x ∈ 𝕏` with `𝕏` the domain of `J`. It can be
noted that the following property must hold `∀ α ∈ ℝ`: `x ∈ 𝕏` implies that `α*x ∈ 𝕏`. In
other words, `𝕏` must be a cone.

Typically, `AMORS` is suitable to solve estimation problems where the unknowns, `x` and
`y`, are the components of a bilinear model given some observations of this model and
`G(x⊗y)` is a data-fidelity term (the lower the better is the agreement of the model with
the observations) while `J(x)` and `K(y)` are regularization terms implementing a priori
constraints in the components.


The `AMORS` algorithm is described in:

1. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Exploiting the scaling
   indetermination of bi-linear models in inverse problems*", in 28th European Signal
   Processing Conference (EUSIPCO), pp. 2358–2362 (2021)
   [DOI](https://doi.org/10.23919/Eusipco47968.2020.9287593).

2. Samuel Thé, Éric Thiébaut, Loïc Denis, and Ferréol Soulez, "*Unsupervised
   blind-deconvolution with optimal scaling applied to astronomical data*", in Adaptive
   Optics Systems VIII, International Society for Optics and Photonics (SPIE), Vol. 12185
   (2022) [DOI](https://doi.org/10.1117/12.2630245).
