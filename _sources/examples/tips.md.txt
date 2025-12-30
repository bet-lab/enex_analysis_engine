# Tips and Best Practices

1. **Always call `system_update()`** after modifying parameters and before accessing results.

2. **Check input ranges**: Some parameters have physical limits (e.g., temperatures must be positive, flow rates must be positive).

3. **Unit consistency**: Be aware of input units (typically °C for temperatures, L/min for flow rates) and output units (typically K for temperatures, W for power).

4. **Iterative models**: Ground-source heat pump models use iterative solvers. If convergence issues occur, check input parameters.

5. **Balance verification**: Energy balances should satisfy conservation (in ≈ out), while entropy and exergy balances include generation/consumption terms.

6. **Performance**: For parameter studies, consider caching results or using vectorized operations where possible.

7. **Visualization**: Use `dartwork-mpl` for publication-quality plots as shown in the examples.
