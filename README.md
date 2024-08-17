# findfunc

findfunc takes as input a txt file where each line is a mathematical equation, and returns the 2D function that best satisfies all equations. It is similar to [PINNs](https://github.com/maziarraissi/PINNs) in the way that the given equations can be PDEs. Under the hood, a [SIREN network](https://arxiv.org/abs/2006.09661) is trained to minimize a set of losses formulated from the different equations. It also implements a simple parser of mathematical expressions to check the validity of each equation.

Requirements:
- [pytorch](https://pytorch.org/)
- [matplotlib](https://matplotlib.org/)

Below are examples of input file together with the generated animation of the training process. See the `examples` folder for more examples.


## Simple

Here the network is tasked to find a function `f` that satisfies the equation `f(x, y) = x * y`:

```
f = x * y
```

```bash
python findfunc.py -i examples/simple.txt
```

<img src="https://github.com/user-attachments/assets/a2ea2089-214d-4bff-be2c-87b64e680503" width="300" height="250"/>


## Heat

Regular math functions and gradient operator can also be thrown into the mix:

```
T(0, y) = sin(4 * 3.14 * y)
grad(T, x) = 0.025 * grad(grad(T, y), y)
```

```bash
python findfunc.py -i examples/heat.txt
```

<img src="https://github.com/user-attachments/assets/ddd6bdeb-02f0-4a20-9e93-40bde02e8426" width="300" height="250"/>


## Navier-Stokes (Lid-driven cavity, Re=100)

Write as many equations as you like: 

```
u(x, 1) = 0
v(x, 1) = 1
u(x, 0) = 0
v(x, 0) = 0
u(0, y) = 0
v(0, y) = 0
u(1, y) = 0
v(1, y) = 0
u * grad(u, x) + v * grad(u, y) + grad(p, x) = 1/100 * (grad(grad(u, x), x) + grad(grad(u, y), y))
u * grad(v, x) + v * grad(v, y) + grad(p, y) = 1/100 * (grad(grad(v, x), x) + grad(grad(v, y), y))
grad(u, x) + grad(v, y) = 0
```

```bash
python findfunc.py -i examples/navier_stokes.txt
```

<img src="https://github.com/user-attachments/assets/1b6657f3-d04f-4486-9022-57ce382e23cc" width="900" height="250"/>

## Image

Finally, images can be part of an equation as well:

```
f = image('starry_night.png', x, y)
```

```bash
python findfunc.py -i examples/starry_night.txt --omega_0 100.0 --hidden_features 512 --hidden_layers 8
```

<img src="https://github.com/user-attachments/assets/8fbb52f4-9356-4d34-85e6-bfde3c07fcc6" width="300" height="250"/>
