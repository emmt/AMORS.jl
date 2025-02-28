# User visible changes in `AMORS` package

- `@public` macro to declare non-exported symbols as *public*, a concept introduced in
  Julia 1.11. The macro does nothing on older Julia versions.


## Version 0.3.1

- `AMORS` is now public.

- Add `autoscale` property and fix iteration counting.

- More properties for `AMORS.Info`.


## Version 0.3.0

- New version of the AMORS algorithm more in-line with the EUSIPCO (2021) paper including
  the notation of the variables. Compared to the 0.2 version, directly scaling the
  variables is avoided in favor of scaling the hyperparameters.

- The API has changed (again) for the object representing the problem.

- The observer method can return a symbolic value other than `:searching` to terminate the
  algorithm.

- The solver returns a structured `info` object with much more information (not just the
  status). See the documentation about `AMORS.Info`. The same object is provided as the
  first argument of the observer (instead of just the iteration counter).

- Iteration counting has changed. The iteration counter, given by `info.iter`, is
  incremented for each update of any component of the model. The initial updates on one of
  the components to initialize `α` count as one iteration. The total number of updates is
  given by `info.eval`.

- `AMORS.observer` provides an example of observer that can be used by the solvers.

- The observer also takes an i/o stream as argument which can be specified by the `io`
  keyword in the solvers.


## Version 0.2.1

- The `do_not_scale` keyword has been replaced by an `autoscale` keyword with opposite
  meaning.

- The `atol` keyword has been renamed `αtol`.


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
