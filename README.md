# findfunc

findfunc implements a super simple framework to quickly find the function that satifisfies a set of equations using a neural network. It is similar to PINNs in the way the given equations can be PDEs. Under the hood, a SIREN network is trained to minimize a set of losses formulated from the different equations.

## Simple
Below is an example of a simple config file where the network is trained to fit the function `f(x, y) = x * y`
```yaml
unknown:
  - f
equations:
  domain:
    - f = x * y
```
`python findfunc.py --config config/simple.yaml --output_file out.gif`


## Heat
```yaml
unknown:
  - f
equations:
  left:
    - f = sin(4 * 3.14 * y)
  domain:
    - grad(f, x) = 0.025 * grad(grad(f, y), y)
```
`python findfunc.py --config config/heat.yaml --output_file out.gif`


## Poisson
```yaml
unknown:
  - f
equations:
  top:
    - f = 0
  bottom:
    - f = 0
  left:
    - f = 0
  right:
    - f = 0
  domain:
    - grad(grad(f, x), x) + grad(grad(f, y), y) = - 2 * 3.14 ** 2 * sin(3.14 * x) * sin(3.14 * y)
```
`python findfunc.py --config config/poisson.yaml --output_file out.gif`


## Helmholtz
`python findfunc.py --config config/helmholtz.yaml --output_file out.gif`


## Navier stokes (Lid-driven cavity)
`python findfunc.py --config config/navier_stokes.yaml --output_file out.gif`