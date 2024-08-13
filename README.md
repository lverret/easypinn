# findfunc

findfunc implements a super simple framework to quickly find the 2D function that satifisfies a set of equations using a neural network. It is similar to [PINNs](https://github.com/maziarraissi/PINNs) in the way the given equations can be PDEs. Under the hood, a [SIREN network](https://arxiv.org/abs/2006.09661) is trained to minimize a set of losses formulated from the different equations. numpy, matplotlib and pytorch are the only requirements.

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

![poisson](https://github.com/user-attachments/assets/17e738dc-0918-46bc-b0b4-b1085cc990b8)

## Helmholtz
`python findfunc.py --config config/helmholtz.yaml --output_file out.gif`

![helmholtz](https://github.com/user-attachments/assets/d799a41d-3882-49b2-840e-a7b2ebca5ed9)

## Navier stokes (Lid-driven cavity, Re=100)
`python findfunc.py --config config/navier_stokes.yaml --output_file out.gif`

![navier_stokes](https://github.com/user-attachments/assets/4d9ed268-6632-4945-b363-9f2e856c5545)

Training hyperparameters can be tweaked if the found function is not good enough, see the argparse.