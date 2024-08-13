# findfunc

findfunc implements a super simple framework for training a neural network to find the function that satifisfies a set of equations. It is similar to PINNs in the way the given equations can be PDEs. Under the hood, it formulates the different equations as a set of losses and trains a SIREN network to approximate the solution of the equations by minimizing the losses.

## Simple
Below is an example of configuration file where the network is trained to fit the simple function `f(x, y) = x * y`
```yaml
unknown:
  - f
equations:
  domain:
    - f = x * y
```
`python findfunc.py --config config/simple.yaml`
