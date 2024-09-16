# User visible changes in `AMORS` package

## Version 0.2.0

The API has been simplified and type stability has been improved. This introduces a number
of breaking changes:

- Out-of-place solving of the problem can done by calling `AMORS.solve`.

- The scaling factor `α` and the hyperparameters are no longer explicitly specified in the
  API.

- The object `f` implementing the objective function is called as `f(Val(:x),x,y)` to
  yield the best `y` given `x`, as `f(Val(:y),x,y)` to yield the best `x` given `y`,
  and as `f(Val(:alpha),x,y)` to yield the best scaling factor which can be computed by
  calling `AMORS.best_scaling_factor`.

- A limited number of methods are assumed to applicable to the variables, say `x`, of the
  problem: `AMORS.scale!(α::Real,x)` to scale in-place the variables,
  `AMORS.has_converged(x,xp,tol)` to check for the convergence in the variables,
  `Base.similar(x)` to allocate a new instance of the variables, and `Base.copyto!(xp,x)`
  to copy the entries of the variables. Methods `AMORS.scale!` and `AMORS.has_converged`
  can be extended for non-arrays variables. The others are basic methods which keep their
  semantic.

- For more flexibilty, another method than `AMORS.has_converged` may be chosen by
  sepcifying keyword `has_converged` in `AMORS.solve` and `AMORS.solve!`.
