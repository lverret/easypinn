# findfunc

findfunc implements a super simple framework to quickly find the 2D function that satifisfies a set of equations using a neural network. It is similar to [PINNs](https://github.com/maziarraissi/PINNs) in the way that the given equations can be PDEs. Under the hood, a [SIREN network](https://arxiv.org/abs/2006.09661) is trained to minimize a set of losses formulated from the different equations.

Requirements:
- [pytorch](https://pytorch.org/)
- [matplotlib](https://matplotlib.org/)

## Simple
Below is an example of a simple config file where the network is trained to fit the function `f(x, y) = x * y`.

```yaml
unknown:
  - f
equations:
  domain:
    - f = x * y
```
`python findfunc.py --config config/simple.yaml --output_file out.gif`

![simple](https://github.com/user-attachments/assets/86c46751-a743-43f2-89e8-0353adf17b46)

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

![heat](https://github.com/user-attachments/assets/9cf10dd7-1600-4789-b2f4-7526181343c7)

## Burger
```yaml
unknown:
  - f
equations:
  left:
    - f = sin(2 * 3.14 * y)
  top:
    - f = 0
  bottom:
    - f = 0
  domain:
    - grad(f, x) + 0.5 * f * grad(f, y) - 0.25 * (0.01 / 3.14) * grad(grad(f, y), y) = 0
```
`python findfunc.py --config config/burger.yaml --output_file out.gif`

## Helmholtz
`python findfunc.py --config config/helmholtz.yaml --output_file out.gif`

![helmholtz](https://github.com/user-attachments/assets/d799a41d-3882-49b2-840e-a7b2ebca5ed9)

## Schrödinger
`python findfunc.py --config config/schrodinger.yaml --output_file out.gif`


## Navier-Stokes (Lid-driven cavity, Re=100)
`python findfunc.py --config config/navier_stokes.yaml --output_file out.gif`

![navier_stokes](https://github.com/user-attachments/assets/4d9ed268-6632-4945-b363-9f2e856c5545)
