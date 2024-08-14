# findfunc

findfunc implements a super simple framework to quickly find the 2D function that satifisfies a set of equations using a neural network. It is similar to [PINNs](https://github.com/maziarraissi/PINNs) in the way that the given equations can be PDEs. Under the hood, a [SIREN network](https://arxiv.org/abs/2006.09661) is trained to minimize a set of losses formulated from the different equations.

Requirements:
- [pytorch](https://pytorch.org/)
- [matplotlib](https://matplotlib.org/)

Below are examples of config files together with the generated animation. More config can be found in the `config` folder.


## Simple
Here the network is trained to fit the function `f(x, y) = x * y`.

```yaml
unknown:
  - f
equations:
  domain:
    - f = x * y
```

```bash
python findfunc.py --config config/simple.yaml
```

<img src="https://github.com/user-attachments/assets/a2ea2089-214d-4bff-be2c-87b64e680503" width="300" height="250"/>


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

```bash
python findfunc.py --config config/heat.yaml
```

<img src="https://github.com/user-attachments/assets/ddd6bdeb-02f0-4a20-9e93-40bde02e8426" width="300" height="250"/>


## Navier-Stokes (Lid-driven cavity, Re=100)

```yaml
unknown:
  - u
  - v
  - p
equations:
  top:
    - u = 0
    - v = 1
  bottom:
    - u = 0
    - v = 0
  left:
    - u = 0
    - v = 0
  right:
    - u = 0
    - v = 0
  domain:
    - u * grad(u, x) + v * grad(u, y) + grad(p, x) = 1/100 * (grad(grad(u, x), x) + grad(grad(u, y), y))
    - u * grad(v, x) + v * grad(v, y) + grad(p, y) = 1/100 * (grad(grad(v, x), x) + grad(grad(v, y), y))
    - grad(u, x) + grad(v, y) = 0
```

```bash
python findfunc.py --config config/navier_stokes.yaml
```

<img src="https://github.com/user-attachments/assets/1b6657f3-d04f-4486-9022-57ce382e23cc" width="900" height="250"/>

## Image

```yaml
unknown:
  - f
equations:
  domain: 
    - f = image('starry_night.png', x, y)
```

```bash
python findfunc.py --config config/starry_night.yaml --omega_0 100.0 --hidden_features 512 --hidden_layers 8
```

<img src="https://github.com/user-attachments/assets/8fbb52f4-9356-4d34-85e6-bfde3c07fcc6" width="300" height="250"/>
