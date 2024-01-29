import argparse
import os

try:
    from tqdm import tqdm
    USE_TQDM=True
except ModuleNotFoundError:
    USE_TQDM=False

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD=True
except ModuleNotFoundError:
    print("Warning! Tensorboard not available")
    USE_TENSORBOARD=False

from src.model.gaussians import Gaussians,RenderPackage
from src.data.colmap import ColmapDataSet
from src.utils.loss_utils import L1Loss,DSSIMLoss,CombinedLoss
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams

def train_report(
    iter:int, # !first iter is 1, not 0!
    model:Gaussians,
    dataset:ColmapDataSet,
    device,
    train_args:TrainParams,
    rendering_pkg:RenderPackage,
    loss:float,
    pbar:'tqdm'=None,
    summarizer:'SummaryWriter'=None,
):
    # Update progress bar
    if pbar is not None and iter%10 == 0:
        pbar.set_postfix({
            'loss':f"{loss:.2e}",
            '#splats': f"{model.num_points:.2e}",
        }, refresh=False)
        pbar.update(10)

    elif pbar is None and iter%(train_args.iterations//100) == 0:
        print("#",end="",flush=True)

    # Update summarizer
    if summarizer is not None:
        summarizer.add_scalar("Loss/Train", loss, iter)
        summarizer.add_scalar("Model/Splats", model.num_points, iter)

        if iter%100 == 0:
            grads = ( torch.linalg.norm(model._xys_grad_accum, dim=1) / model._xys_grad_norm )
            grads[grads.isnan()] = 0.
            summarizer.add_histogram("Dev/2DGrads", grads.detach().cpu().numpy(), iter)

    if iter in train_args.save_at:
        # TODO save model
        pass

    if iter in train_args.test_at:
        # TODO run test iter
        pass

def train_loop(
    model:Gaussians,
    dataset:ColmapDataSet,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams
):

    # Set up optimizer
    # From https://github.com/nerfstudio-project/gsplat/issues/87#issuecomment-1862540059
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            # 10 warmup iters
            torch.optim.lr_scheduler.LinearLR(
                model.optimizer, start_factor=0.01, total_iters=10
            ),
            # exponential decay until a factor 0.1
            torch.optim.lr_scheduler.MultiStepLR(
                model.optimizer,
                milestones=[
                    train_args.iterations * 1 // 6,
                    train_args.iterations * 2 // 6,
                    train_args.iterations * 3 // 6,
                    train_args.iterations * 4 // 6,
                    train_args.iterations * 5 // 6,
                ],
                gamma=0.1 ** (1/5),
            ),
        ]
    )

    # Set up loss
    loss_fn = CombinedLoss(
        ( L1Loss(), 1.-train_args.lambda_dssim ),
        ( DSSIMLoss(device=device), train_args.lambda_dssim )
    )

    # Use TQDM progress bar
    pbar = None
    if USE_TQDM:
        pbar = tqdm(total=train_args.iterations, desc="Training", smoothing=.5)

    # Use Tensorboard
    summarizer = None
    if USE_TENSORBOARD:
        print("Tensorboard running at", pipeline_args.log_dir)
        summarizer = SummaryWriter(log_dir=pipeline_args.log_dir)

    dataset_cycle = dataset.cycle()

    for iter in range(1,train_args.iterations+1):

        camera = next(dataset_cycle)
        camera.to(device)

        model.optimizer.zero_grad()

        # Forward pass
        rendering_pkg = model.render(camera)

        loss = loss_fn(rendering_pkg.rendering, camera.gt_image)

        # Backward pass
        loss.backward()
        model.optimizer.step()

        with torch.no_grad():

            # Densify
            if train_args.densify_from < iter <= train_args.densify_until:

                model.update_densification_stats(
                    xys=rendering_pkg.xys,
                    radii=rendering_pkg.radii,
                    visibility_mask=rendering_pkg.visibility_mask
                )

                if (iter-train_args.densify_from) % train_args.densify_every == 0:
                    model.densify(
                        grad_threshold=train_args.grad_threshold,
                        max_scale=train_args.max_density * dataset.scene_extend,
                        min_opacity=train_args.min_opacity,
                    )

            # Reset opacity
            if train_args.reset_opacity_from < iter <= train_args.reset_opacity_until \
            and (iter - train_args.reset_opacity_from) % train_args.reset_opacity_every == 0:
                model.reset_opacity()

            # Report on iter
            train_report(
                iter=iter,
                model=model,
                dataset=dataset,
                device=device,
                train_args=train_args,
                rendering_pkg=rendering_pkg,
                loss=loss,
                pbar=pbar,
                summarizer=summarizer
            )

        # End iter
        scheduler.step()
        camera.to('cpu')

    if pbar is not None:
        pbar.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
    dp = DataParams(parser)
    mp = ModelParams(parser)
    tp = TrainParams(parser)
    pp = PipeLineParams(parser)

    args = parser.parse_args()

    data_args = dp.extract(args)
    model_args = mp.extract(args)
    train_args = tp.extract(args)
    pipeline_args = pp.extract(args)

    if pipeline_args.no_pbar:
        USE_TQDM = False

    try:
        torch.cuda.set_device(args.device)
    except:
        pass

    dataset = ColmapDataSet(device=args.device, **vars(data_args))

    if args.source_path[:-1].endswith("3du_data_"):
        print("Initializing model randomly with {} points in an area of size {:.1f}".format(
            model_args.num_points,
            dataset.scene_extend
        ))
        model = Gaussians(
            device=args.device,
            scene_extend=dataset.scene_extend,
            **vars(model_args),
        )
    else:
        fname = os.path.join(args.source_path,'sparse','0','points3D.txt')
        print(f"Initializing model from {fname}")
        model = Gaussians.from_colmap(
            path_to_points3D_file=fname,
            device=args.device,
            scene_extend=dataset.scene_extend,
            **vars(model_args),
        )

    train_loop(
        model=model,
        dataset=dataset,
        device=args.device,
        train_args=train_args,
        pipeline_args=pipeline_args
    )
