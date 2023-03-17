import os
import argparse

import torch
from torch_em.data.datasets import get_livecell_loader

from prob_utils.my_utils import DummyLoss
from prob_utils.my_models import ProbabilisticUnet
from prob_utils.my_predictions import punet_prediction
from prob_utils.my_evaluations import run_dice_evaluation
from prob_utils.my_trainer import PUNetTrainer, PUNetLogger

def get_livecell_loaders(path:str, ctype:list, patch_shape=(512, 512)):

    train_loader = get_livecell_loader(
                        path=path, 
                        binary=True, 
                        split='train', 
                        patch_shape = patch_shape, 
                        batch_size=4, 
                        cell_types=[ctype],
                        download=True
                    )
    
    val_loader = get_livecell_loader(
                        path=path, 
                        binary=True, 
                        split='val', 
                        patch_shape = patch_shape, 
                        batch_size=1, 
                        cell_types=[ctype],
                        download=True
                    )
    
    return train_loader, val_loader

def do_punet_training(device, data_path:str):
    cell_types_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']
    
    for ctype in cell_types_list:

        os.makedirs(data_path, exist_ok=True)

        train_loader, val_loader = get_livecell_loaders(path=data_path, ctype=ctype)

        model = ProbabilisticUnet(
                    input_channels=1, 
                    num_classes=1, 
                    num_filters=[64,128,256,512], 
                    latent_dim=6, 
                    no_convs_fcomb=3, 
                    beta=1.0, 
                    rl_swap=True
                )
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
        
        trainer = PUNetTrainer(
                    name=f"punet-source-livecell-{ctype}",
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    optimizer=optimizer,
                    loss=DummyLoss(),  
                    metric=DummyLoss(),  
                    device=device,
                    lr_scheduler=scheduler,
                    logger=PUNetLogger,
                    mixed_precision=True,  
                    log_image_interval=1000
                )

        n_iterations = 100000
        trainer.fit(n_iterations)

def do_punet_predictions(device, data_path:str, pred_path:str):

    cell_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    model = ProbabilisticUnet(
                input_channels=1,
                num_classes=1,
                num_filters=[64,128,256,512],
                latent_dim=6,
                no_convs_fcomb=3,
                beta=1.0,
                rl_swap=True
            )

    for ctype1 in cell_list:
        model_save_dir = f"checkpoints/punet-source-livecell-{ctype1}/best.pt"

        if os.path.exists(model_save_dir) is False:
            print("The source model couldn't be found/hasn't been trained yet")
            continue

        model_state = torch.load(model_save_dir, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(model_state)
        model.to(device)

        for ctype2 in cell_list:

            input_path = data_path + f"images/livecell_test_images/{ctype2}*"
            output_path = pred_path + f"punet_source/{ctype1}/{ctype2}/"

            punet_prediction(input_image_path=input_path, output_pred_path=output_path, model=model, device=device)

def do_punet_evaluations(data_path:str, pred_path:str):

    cell_types_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    for ctype1 in cell_types_list:
        gt_dir = data_path + f"annotations/livecell_test_images/{ctype1}/"

        for ctype2 in cell_types_list:
            pred_dir = pred_path + f"punet_source/{ctype2}/{ctype1}/"

            if os.path.exists(pred_dir) is False:
                print("The source model predictions couldn't be found/haven't been generated")
                continue

            run_dice_evaluation(gt_dir, pred_dir)

            print(f"Dice for Target Cells - {ctype1} from Source Cells - {ctype2}")

def main(args):
    try:
        print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    except AssertionError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D PUNet on LiveCELL dataset")
        do_punet_training(data_path=args.data, device=device)

    if args.predict:
        print("Getting predictions on LiveCELL dataset from the trained PUNet")
        do_punet_predictions(data_path=args.data, pred_path=args.pred_path, device=device)

    if args.evaluate:
        print("Evaluating the PUNet predictions of LiveCELL dataset")
        do_punet_evaluations(data_path=args.data, pred_path=args.pred_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action='store_true', help="Enables PUNet training on LiveCELL dataset")
    parser.add_argument("--predict", action='store_true', help="Obtains PUNet predictions on LiveCELL test-set")
    parser.add_argument("--evaluate", action='store_true', help="Evaluates PUNet predictions")

    parser.add_argument("--data", type=str, default="~/data/livecell/", help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("--pred_path", type=str, default="~/predictions/livecell/", help="Path where predictions will be saved")

    args = parser.parse_args()
    main(args)