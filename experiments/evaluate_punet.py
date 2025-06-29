import h5py

import torch

from common import get_dataloaders, get_model


def evaluate_punet(data_path, dataset_name, patch_shape, save_root, **kwargs):
    """Evaluate PUNet model.
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    _, val_loader = get_dataloaders(
        data_path=data_path, dataset_name=dataset_name, patch_shape=patch_shape, **kwargs
    )
    model = get_model(dataset_name=dataset_name, device=device, backbone="punet", testing=True)
    model.load_state_dict(
        torch.load("checkpoints/punet-source-em/best.pt", map_location="cpu", weights_only=False)["model_state"]
    )
    model.eval()

    activation = torch.nn.Sigmoid()
    n_samples = 2

    for i, (x, y) in enumerate(val_loader):
        model(x.to(device))
        outputs = [activation(model.sample()) for _ in range(n_samples)]
        avg_pred = torch.stack(outputs, dim=0).sum(dim=0) / n_samples
        avg_pred = avg_pred.squeeze().detach().cpu().numpy()

        with h5py.File(f"res_{i}.h5", "w") as f:
            f.create_dataset("image", data=x.squeeze().numpy(), compression="gzip")
            f.create_dataset("gt", data=y.squeeze().numpy(), compression="gzip")
            f.create_dataset("prediction", data=avg_pred, compression="gzip")

        breakpoint()


def main(args):
    evaluate_punet(args.input_path, "em", patch_shape=(16, 512, 512), save_root=None, subtypes="mitoem")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/data")
    args = parser.parse_args()
    main(args)
