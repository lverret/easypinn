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

![simple](https://github.com/user-attachments/assets/a2ea2089-214d-4bff-be2c-87b64e680503)

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

![heat](https://github.com/user-attachments/assets/ddd6bdeb-02f0-4a20-9e93-40bde02e8426)

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

![burger](https://github.com/user-attachments/assets/883c68d4-5124-4e56-a1ab-b9b0c7152efe)

## Helmholtz
`python findfunc.py --config config/helmholtz.yaml --output_file out.gif`

![helmholtz](https://github.com/user-attachments/assets/d8180997-3e8c-457e-9a8f-75b9c6a48f7f)

## Schrödinger
`python findfunc.py --config config/schrodinger.yaml --output_file out.gif`

![schrodinger](https://github.com/user-attachments/assets/7f23916b-0943-43d7-828a-b9582480bad3)


## Navier-Stokes (Lid-driven cavity, Re=100)
`python findfunc.py --config config/navier_stokes.yaml --output_file out.gif`

![navier_stokes](https://github.com/user-attachments/assets/1b6657f3-d04f-4486-9022-57ce382e23cc)

