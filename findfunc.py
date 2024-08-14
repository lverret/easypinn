import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import trange
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter


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

    def forward(self, xy_list):
        return [self.net(torch.cat(xy, dim=-1)) for xy in xy_list]


# -----------------------------------------------------------------------------
# Helper mathematical functions for defining the equations to solve


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


def grad(y, x):
    return torch.autograd.grad(
        y, [x], grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


# -----------------------------------------------------------------------------
# Training and visualization functions


def generate_samples(equations):
    xy_list = []
    for location in equations:
        if location == "top":
            x = torch.rand((args.nb_samples, 1))
            y = torch.ones((args.nb_samples, 1))
        elif location == "bottom":
            x = torch.rand((args.nb_samples, 1))
            y = torch.zeros((args.nb_samples, 1))
        elif location == "left":
            x = torch.zeros((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        elif location == "right":
            x = torch.ones((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        elif location in ["domain", "image"]:
            x = torch.rand((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        x = x.clone().detach().requires_grad_(True).to(args.device)
        y = y.clone().detach().requires_grad_(True).to(args.device)
        xy_list.append((x, y))
    return xy_list


def compute_loss(out_list, xy_list, unknown, equations):
    loss = 0
    res = list(equations.values())
    for out, (x, y), location in zip(out_list, xy_list, equations):
        for k, var in enumerate(unknown):
            exec(f"{var}=out[:, k:k+1]")
        if location == "image":
            assert len(unknown) == 1, "Image only supports 1 unknown"
            assert len(equations[location]) == 1, "Image only supports 1 equation"
            image = np.array(Image.open(equations[location][0]).convert("L"))
            image = torch.tensor(image).rot90(-1) / 255
            image = image.to(out.device)
            px, py = ((x * image.size(0)).long(), (y * image.size(1)).long())
            loss += torch.mean(torch.abs(out - image[px, py]))
        else:
            if location == "domain":
                w = 0.1
            else:
                w = 0.9
            for res in equations[location]:
                splits = res.split("=")
                assert len(splits) == 2, "Incorrect definition of equations"
                lhs, rhs = splits
                loss += w * torch.mean(torch.abs(eval(f"{lhs} - ({rhs})")))
    return loss


def make_gif(frames, vars):
    ims = []
    fig, axs = plt.subplots(1, len(vars), figsize=(4.8 * len(vars), 4.0))
    if len(vars) == 1:
        axs = (axs,)
    for k, var in enumerate(vars):
        ims.append(axs[k].imshow(frames[0][:, :, k], extent=(0, 1, 0, 1)))
        fig.colorbar(ims[k], ax=axs[k])
        axs[k].set_xlabel("x")
        axs[k].set_ylabel("y")
        axs[k].set_title(var)
        axs[k].margins(0)

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
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--output_file", type=str, help="output filename for the gif")
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

    with open(f"{args.config}", "r") as fd:
        config = yaml.safe_load(fd)

    unknown, equations = config["unknown"], config["equations"]

    model = Siren(
        in_features=2,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=len(unknown),
        first_omega_0=args.omega_0,
        hidden_omega_0=30.0,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    pbar = trange(args.nb_iter, desc="Finding function")
    frames = []

    for it in pbar:
        xy_list = generate_samples(equations)
        out_list = model(xy_list)
        loss = compute_loss(out_list, xy_list, unknown, equations)
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
            xy_test = [(xy[:, 0:1], xy[:, 1:2])]
            out = model(xy_test)[0]
            out = out.view(args.resolution, args.resolution, out.size(-1))
            out = out.rot90().cpu().data.numpy()
            frames.append(out)

    make_gif(frames, unknown)
