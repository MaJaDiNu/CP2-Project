# Exercise 7 observations

The simulations use 100,000 steps (`dt = 0.005`) for a total reduced time of
500. Energy is conserved well: the largest relative deviations from the
initial energy are about `8.3e-4` for `T0 = 0.1` and `2.0e-4` for `T0 = 1.5`.
Total momentum remains below `7.6e-13` in both runs, consistent with numerical
round-off.

For `T0 = 0.1`, the potential energy per particle decreases from zero to an
average of about `-0.406` over the final 20% of the run, while the mean
temperature rises to about `0.370`. The initially dilute particles therefore
form locally dense, bound clusters. The released binding energy becomes kinetic
energy; this explains why the temperature rises while total energy stays
constant.

For `T0 = 1.5`, the final mean potential energy per particle is only about
`-0.027`, and the temperature stays close to its initial value (about `1.518`).
Thermal motion is strong enough to prevent persistent aggregation, so this run
remains a dilute gas with only brief pair encounters.

The radial distribution functions support this interpretation. Both runs show
the excluded core below approximately `r = 1`. The cold run has a very strong
nearest-neighbour peak near the Lennard-Jones minimum (the sampled maximum is
`g(1.15) = 27.7`) and additional short-range structure, characteristic of dense
clusters. The hot run has only a low, broad first peak (`g(1.15) = 1.75`) and
quickly approaches `g(r) = 1`, as expected for a dilute, weakly correlated gas.
