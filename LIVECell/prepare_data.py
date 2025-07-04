import argparse

from torch_em.data.datasets.livecell import _download_livecell_images, _download_livecell_annotations


def download_livecell_data(path):
    # download the train / val / test data and create the segmentations
    _download_livecell_images(path, download=True)
    _download_livecell_annotations(path, split="train", download=True, cell_types=None, label_path=None)
    _download_livecell_annotations(path, split="test", download=True, cell_types=None, label_path=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="~/data/livecell/",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )
    args = parser.parse_args()
    download_livecell_data(args.data)


if __name__ == "__main__":
    main()
