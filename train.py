import os
from tqdm.notebook import tqdm
from random import shuffle
import datetime

import matplotlib.pyplot as plt
import numpy as np

import torch
import gsplat
from PIL import Image
from torch import optim
import torchvision.transforms as transforms

from src.model.gaussians import Gaussians
from src.data.colmap import ColmapDataSet
from src.data.customdataset import CustomDataSet
from src.utils.loss_utils import L1Loss,DSSIMLoss,CombinedLoss


def image_path_to_tensor(image_path:str):

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def tensor_to_ndarray(image:torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()

def ndarray_to_Image(image:np.ndarray) -> Image.Image:
    return Image.fromarray((image*255).astype(np.uint8))

def tensor_to_Image(image:torch.Tensor) -> Image.Image:
    return ndarray_to_Image(tensor_to_ndarray(image))


def train(
    model:Gaussians,
    dataset:ColmapDataSet,
    device=DEVICE,
    num_iterations:int=7_000,
) -> tuple[Gaussians,dict]:
    # Set up optimizer
    # From https://github.com/nerfstudio-project/gsplat/issues/87#issuecomment-1862540059
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            # 10 warmup iters
            torch.optim.lr_scheduler.LinearLR(
                model.optimizer, start_factor=0.01, total_iters=10
            ),
            # multiply lr by 1/3 at 50% and 75%
            torch.optim.lr_scheduler.MultiStepLR(
                model.optimizer,
                milestones=[
                    num_iterations // 2,
                    num_iterations * 3 // 4,
                ],
                gamma=0.33,
            ),
        ]
    )

    lambda_dssim = .2
    loss_fn = CombinedLoss( (L1Loss(), 1.-lambda_dssim), (DSSIMLoss(), lambda_dssim) )

    # Output package
    out = {
        'losses':[],
        'lr':[]
    }

    # Use progress bar
    pbar = tqdm(total=num_iterations, desc="Training", smoothing=.5)

    test_cam = dataset.cameras[1]
    test_cam.to(device)
    tensor_to_Image(test_cam.gt_image).save(f'renders/gt_{test_cam.name}')

    # Set up epoch
    loss_accum=0.
    loss_norm=0

    dataset_cycle = dataset.cycle()

    for iter in range(1,num_iterations+1):

        camera = next(dataset_cycle)
        camera.to(device)

        model.optimizer.zero_grad()

        # Forward pass
        rendering_pkg = model.render(camera)

        loss = loss_fn(rendering_pkg['rendering'], camera.gt_image)

        # Backward pass
        loss.backward()
        model.optimizer.step()

        with torch.no_grad():

            if iter <= 15_000:
                # Densify
                model.update_densification_stats(
                    xys=rendering_pkg['xys'],
                    radii=rendering_pkg['radii'],
                    visibility_mask=rendering_pkg['visibility_mask']
                )
                if iter >= 1000 and iter%500 == 0:
                    model.densify(
                        grad_threshold=2e-8,
                        max_scale=0.01 * dataset.scene_extend,
                        min_opacity=0.005,
                    )

                if iter%3000 == 0:
                    model.reset_opacity()

            # Update batch info
            loss_accum += loss.item()
            loss_norm += 1

            pbar.update(1)

            if iter%10 == 0:
                # Compute epoch loss
                _loss = loss_accum / loss_norm
                _lr = scheduler.get_last_lr()[0]

                out['losses'].append(_loss)
                out['lr'].append(_lr)
                loss_accum = 0.
                loss_norm = 0

                pbar.set_postfix({
                    'loss':f"{_loss:.2e}",
                    'lr':f"{_lr:.1e}",
                    '#splats': model.num_points,
                }, refresh=False)

            if (iter-1)%100 == 0:
                test_cam.to(device)
                # tensor_to_Image(model.render(camera, bg=torch.zeros(3,device=device))).save(f'renders/epoch{epoch}_{camera.name}')
                tensor_to_Image(model.render(test_cam, bg=torch.zeros(3,device=device))['rendering']).save(f'renders/latest_{test_cam.name}')
                test_cam.to('cpu')

        # End iter
        scheduler.step()
        camera.to('cpu')
        torch.cuda.empty_cache()

    pbar.close()

    return out

if __name__ == '__main__':
    ROOT_DIR = '/media/jip/T7/thesis/code/data/'
    if not os.path.isdir(ROOT_DIR):
        ROOT_DIR = '/home/jip/data1/'
    ROOT_DIR += "3du_data_2"

    DEVICE = 'cuda:0'
    try:
        torch.cuda.set_device(DEVICE)
    except:
        print("!WARNING! could not set cuda device, falling back to default")


    if ROOT_DIR[:-1].endswith("3du_data_"):
        dataset = CustomDataSet(
            root_dir=ROOT_DIR,
            img_folder='images_8',
            device='cpu',
        )
    else:
        dataset = ColmapDataSet(
            root_dir=ROOT_DIR,
            img_folder='images_8',
            device='cpu',
        )
    print(f"Found {len(dataset)} cameras")

    model = Gaussians(
        num_points=100_000,
        scene_extend=dataset.scene_extend,
        device=DEVICE,
    )
    print("Model initialized with {} points".format(model.num_points))

    out = train(
        model,
        dataset,
        num_iterations=7_000,
        device=DEVICE,
    )

    fname = "models/model_{}.ckpt".format(datetime.datetime.now().strftime('%a%d%b%H%M').lower())
    torch.save(model.state_dict(),fname)
