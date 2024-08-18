import argparse
import ast
import numpy as np
import torch
import operator
import matplotlib.pyplot as plt

from inspect import signature
from tqdm import trange
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter


class InvalidFormula(Exception):
    def __init__(self, formula, reason, *args):
        self.message = f"'{formula}' ({reason})"
        super(InvalidFormula, self).__init__(self.message, *args)


# -----------------------------------------------------------------------------
# Neural architectures


class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
        else:
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.in_features) / self.omega_0,
                np.sqrt(6 / self.in_features) / self.omega_0,
            )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=3.0,
        hidden_omega_0=3.0,
    ):
        super().__init__()
        self.net = [
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        ]
        for _ in range(hidden_layers):
            self.net.append(
                (
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=hidden_omega_0,
                    )
                )
            )
        final_linear = torch.nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )
        self.net.append(final_linear)
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


# -----------------------------------------------------------------------------
# Helper mathematical functions for defining the equations to solve


def sqrt(y):
    return torch.sqrt(torch.as_tensor(y))


def sin(y):
    return torch.sin(torch.as_tensor(y))


def cos(y):
    return torch.cos(torch.as_tensor(y))


def exp(y):
    return torch.exp(torch.as_tensor(y))


def abs(y):
    return torch.abs(torch.as_tensor(y))


def tanh(y):
    return torch.tanh(torch.as_tensor(y))


def image(path, x, y):
    image = np.array(Image.open(path).convert("L"))
    image = torch.tensor(image).rot90(-1) / 255
    image = image.to(x.device)
    px, py = ((x * image.size(0)).long(), (y * image.size(1)).long())
    return image[px, py]


def grad(y, x):
    return torch.autograd.grad(
        y, [x], grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


OPS = {
    ast.USub: operator.neg,
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


FUNS = {
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "exp": exp,
    "abs": abs,
    "tanh": tanh,
    "image": image,
    "grad": grad,
}

MODEL = {}

# -----------------------------------------------------------------------------
# Parser functions


def parse_equations(equations):
    vars = {}
    domains = []
    for formula in equations:
        splits = formula.split("=")
        if len(splits) != 2:
            raise InvalidFormula(formula, "Not a equation")
        lhs, rhs = splits
        domain = {"x": torch.nan, "y": torch.nan}
        vars |= parse(formula, ast.parse(lhs.strip(), mode="eval").body, vars, domain)
        vars |= parse(formula, ast.parse(rhs.strip(), mode="eval").body, vars, domain)
        domains.append(domain)
        log = f"Parsed equation {len(domains)}: 'for "
        for inp in ["x", "y"]:
            if np.isnan(domain[inp]):
                log += f"{inp} in (0, 1), "
            else:
                log += f"{inp} = {domain[inp]}, "
        print(log[:-2] + f",  {formula}'")
    vars = list(vars.keys())
    for k, var in enumerate(vars):
        code = f"global {var}\n" f"def {var}(x, y): return model(x, y)[:, {k}:{k+1}]"
        exec(compile(code, "", "exec"))
        MODEL[var] = globals()[var]
    print(f"Found {len(vars)} unknown fonction(s) to approximate: {vars}")
    return vars, domains


def parse(formula, node, vars, domain):
    if isinstance(node, ast.Constant):
        return vars
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return parse(formula, node.operand, vars, domain)
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        vars |= parse(formula, node.left, vars, domain)
        vars |= parse(formula, node.right, vars, domain)
        return vars
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in FUNS:
            if len(node.args) != len(signature(FUNS[node.func.id]).parameters):
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            for arg in node.args:
                vars |= parse(formula, arg, vars, domain)
            return vars
        elif isinstance(node.func, ast.Name) and node.func.id not in FUNS:
            if not all(isinstance(arg, (ast.Constant, ast.Name)) for arg in node.args):
                raise InvalidFormula(formula, f"Found invalid arg for '{node.func.id}'")
            if len(node.args) != 2:
                raise InvalidFormula(
                    formula, f"Invalid nb of args for '{node.func.id}'"
                )
            if (
                isinstance(node.args[0], ast.Name)
                and node.args[0].id != "x"
                or isinstance(node.args[1], ast.Name)
                and node.args[1].id != "y"
            ):
                raise InvalidFormula(
                    formula, f"'{node.func.id}' takes as args only (x, y) in that order"
                )
            for inp, arg in zip(["x", "y"], node.args):
                if isinstance(arg, ast.Num):
                    if not 0 <= arg.n <= 1:
                        raise InvalidFormula(
                            formula, "Only functions in [0, 1] x [0, 1] are supported"
                        )
                    domain[inp] = float(arg.n)
            vars[node.func.id] = None
            return vars
    elif isinstance(node, ast.Name):
        if node.id in ["x", "y"]:
            return vars
        else:
            vars = parse(
                formula,
                ast.Call(ast.Name(node.id), [ast.Name("x"), ast.Name("y")]),
                vars,
                domain,
            )
            return vars
    raise InvalidFormula(formula, "Found unsupported token(s)")


def eval(node, samples):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](eval(node.operand, samples))
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](eval(node.left, samples), eval(node.right, samples))
    elif isinstance(node, ast.Call) and node.func.id in FUNS:
        return FUNS[node.func.id](*(eval(arg, samples) for arg in node.args))
    elif isinstance(node, ast.Call) and node.func.id in MODEL:
        return MODEL[node.func.id](
            eval(ast.Name("x"), samples), eval(ast.Name("y"), samples)
        )
    elif isinstance(node, ast.Name):
        if node.id in ["x", "y"]:
            return samples[node.id]
        else:
            return eval(
                ast.Call(ast.Name(node.id), [ast.Name("x"), ast.Name("y")]), samples
            )
    raise RuntimeError()


# -----------------------------------------------------------------------------
# Training and visualization functions


def generate_samples(domain):
    samples = {"x": None, "y": None}
    for inp in ["x", "y"]:
        if np.isnan(domain[inp]):
            samples[inp] = torch.rand(args.nb_samples, 1)
        else:
            samples[inp] = torch.ones(args.nb_samples, 1) * domain[inp]
        samples[inp] = (
            samples[inp].clone().detach().requires_grad_(True).to(args.device)
        )
    return samples


def compute_loss(equations, domains):
    loss = 0
    for formula, domain in zip(equations, domains):
        lhs, rhs = formula.split("=")
        samples = generate_samples(domain)
        w = 0.1 if np.isnan(domain["x"]) and np.isnan(domain["y"]) else 0.9
        res = eval(ast.parse(f"{lhs} - ({rhs})".strip(), mode="eval").body, samples)
        loss += w * torch.mean(torch.abs(res))
    return loss


def make_gif(frames, vars):
    ims = []
    nr = int(np.sqrt(len(vars)))
    nc = len(vars) // nr + len(vars) % nr
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 4.0 * nr))
    if nr == 1:
        axs = np.array((axs,))
    if nc == 1:
        axs = np.array((axs,))
    for k, var in enumerate(vars):
        i, j = k // nc, k % nc
        ims.append(axs[i, j].imshow(frames[0][:, :, k], extent=(0, 1, 0, 1)))
        fig.colorbar(ims[k], ax=axs[i, j])
        axs[i, j].set_xlabel("x")
        axs[i, j].set_ylabel("y")
        axs[i, j].set_title(var)
        axs[i, j].margins(0)
    for k in range(len(vars), nc * nr):
        i, j = k // nc, k % nc
        fig.delaxes(axs[i][j])
    fig.tight_layout()

    def animate(i):
        out = frames[i]
        for k in range(out.shape[-1]):
            z = out[:, :, k]
            ims[k].set_data(z)
            ims[k].set_clim(z.min(), z.max())

    ani = FuncAnimation(fig, animate, frames=len(frames))
    pbar = trange(len(frames), desc="Generating GIF")
    ani.save(
        args.output_file,
        writer=PillowWriter(fps=len(frames) / 3),
        progress_callback=lambda i, n: pbar.update(1),
    )


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=str, help="input filename with one equation per line"
    )
    parser.add_argument(
        "--output_file", "-o", type=str, default="out.gif", help="output gif filename"
    )
    parser.add_argument("--nb_iter", type=int, default=500, help="number of iterations")
    parser.add_argument(
        "--nb_samples", type=int, default=1000, help="number of uniform samples"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--hidden_layers", type=int, default=4, help="number of hidden layers"
    )
    parser.add_argument(
        "--hidden_features", type=int, default=256, help="size of the hidden features"
    )
    parser.add_argument(
        "--omega_0", type=float, default=10.0, help="first omega_0 of siren"
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="image resolution for the gif"
    )
    parser.add_argument(
        "--nb_frames", type=int, default=50, help="number of frames for the gif"
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = f.read()
    equations = data.splitlines()

    vars, domains = parse_equations(equations)

    model = Siren(
        in_features=2,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=len(vars),
        first_omega_0=args.omega_0,
        hidden_omega_0=30.0,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    pbar = trange(args.nb_iter, desc="Solving equation(s)")
    frames = []

    for it in pbar:
        loss = compute_loss(equations, domains)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        if it % max(1, (args.nb_iter // args.nb_frames)) == 0:
            xy = torch.cartesian_prod(
                torch.linspace(0, 1, args.resolution),
                torch.linspace(0, 1, args.resolution),
            ).to(args.device)
            out = model(xy[:, 0:1], xy[:, 1:2])
            out = out.view(args.resolution, args.resolution, out.size(-1))
            out = out.rot90().cpu().data.numpy()
            frames.append(out)

    make_gif(frames, vars)
