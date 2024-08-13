import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import warnings

from tqdm import trange

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Neural architectures


class FourierFeatures(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer("B", torch.randn((hidden_dim // 2, input_dim)) * 1)

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B.to(x.device).t()
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_proj


class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.net = []
        for ind in range(hidden_layers):
            layer_dim_in = in_features if ind == 0 else hidden_features
            self.net.append(torch.nn.Linear(layer_dim_in, hidden_features))
            self.net.append(torch.nn.ReLU())
        layer_dim_in = in_features if hidden_layers == 0 else hidden_features
        self.net.append(torch.nn.Linear(layer_dim_in, out_features))
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, xy_list):
        return [self.net(torch.cat(xy, dim=-1)) for xy in xy_list]


class SineLayer(torch.nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
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
        first_omega_0=3,
        hidden_omega_0=3.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )
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
# Helper mathematical functions for defining the PDE


def sin(y):
    return torch.sin(y)


def cos(y):
    return torch.cos(y)


def grad(y, x):
    return torch.autograd.grad(
        y, [x], grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


# -----------------------------------------------------------------------------
# Training functions


def generate_samples(config):
    xy_list = []
    for res in config["equations"]:
        if res == "top":
            x = torch.rand((args.nb_samples, 1))
            y = torch.ones((args.nb_samples, 1))
        elif res == "bottom":
            x = torch.rand((args.nb_samples, 1))
            y = torch.zeros((args.nb_samples, 1))
        elif res == "left":
            x = torch.zeros((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        elif res == "right":
            x = torch.ones((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        elif res == "domain":
            x = torch.rand((args.nb_samples, 1))
            y = torch.rand((args.nb_samples, 1))
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        xy_list.append((x.to(args.device), y.to(args.device)))
    return xy_list


def create_images(out, resolution, config):
    images = []
    for k, var in zip(range(out.size(-1)), config["unknown"]):
        z = out[:, k]
        z = z.view(resolution, resolution).rot90()
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        im = ax.imshow(z.cpu().data.numpy(), extent=(0, 1, 0, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(var)
        fig.colorbar(im)
        ax.margins(0)
        fig.canvas.draw()
        ncols, nrows = fig.canvas.get_width_height()
        z = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        z = z.reshape((nrows, ncols, 3))
        images.append(z)
        plt.close(fig)
    return np.hstack(images)


def compute_loss(out_list, xy_list, config):
    loss = 0
    res = list(config["equations"].values())
    for out, (x, y), type in zip(out_list, xy_list, config["equations"]):
        for k, var in enumerate(config["unknown"]):
            exec(f"{var}=out[:, k:k+1]")
        if type == "domain":
            w = 0.1
        else:
            w = 0.9
        for res in config["equations"][type]:
            splits = res.split("=")
            assert len(splits) == 2, "Incorrect definition of equations"
            lhs, rhs = splits
            loss += w * torch.mean(torch.abs(eval(f"{lhs} - ({rhs})")))
    return loss


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--output_file", type=str, help="output filename for gif")
    parser.add_argument("--nb_iter", type=int, default=500, help="number of iter")
    parser.add_argument(
        "--nb_samples", type=int, default=1000, help="number of uniform samples"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="number of uniform samples"
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="number of uniform samples"
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    args = parser.parse_args()

    with open(f"{args.config}", "r") as fd:
        config = yaml.safe_load(fd)

    model = Siren(
        in_features=2,
        hidden_features=256,
        hidden_layers=4,
        out_features=len(config["unknown"]),
        first_omega_0=10,
        hidden_omega_0=30,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    video = []
    pbar = trange(args.nb_iter, desc="Finding solution")

    for it in pbar:
        xy_list = generate_samples(config)
        out_list = model(xy_list)
        loss = compute_loss(out_list, xy_list, config)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        if it % max(1, (args.nb_iter // 50)) == 0:
            xy = torch.cartesian_prod(
                torch.linspace(0, 1, args.resolution),
                torch.linspace(0, 1, args.resolution),
            ).to(args.device)
            xy_test = [(xy[:, 0:1], xy[:, 1:2])]
            out = model(xy_test)
            images = create_images(out[0], args.resolution, config)
            video.append(images)

    imageio.mimsave(args.output_file, np.array(video), loop=0)
