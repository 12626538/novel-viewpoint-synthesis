import argparse

try:
    from tqdm import tqdm
    USE_TQDM=True
except ModuleNotFoundError:
    USE_TQDM=False

import torch

try:
    from tqdm.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD=True
except ModuleNotFoundError:
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
    pbar:tqdm=None,
    summarizer:SummaryWriter=None,
):
    # Update progress bar
    if pbar is not None and iter%10 == 0:
        pbar.set_postfix({
            'loss':f"{loss:.2e}",
            '#splats': f"{model.num_points:.2e}",
        }, refresh=False)
        pbar.update(10)

    elif pbar is None and iter%(train_args.iterations//100) == 0:
        print("#")

    # Update summarizer
    summarizer.add_scalar("Loss/Train", loss, iter)
    summarizer.add_scalar("Model/Splats", model.num_points, iter)

    if iter%100 == 0:
        summarizer.add_histogram("Dev/2DGrads", model._xys_grad_accum.detach().cpu() / model._xys_grad_norm.detach().cpu(), iter)

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
    train_args:TrainParams
):

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
                    train_args.iterations // 2,
                    train_args.iterations * 3 // 4,
                ],
                gamma=0.33,
            ),
        ]
    )

    # Set up loss
    loss_fn = CombinedLoss(
        ( L1Loss(), 1.-train_args.lambda_dssim ),
        ( DSSIMLoss(), train_args.lambda_dssim )
    )

    # Output package
    out = {
        'losses':[],
        'lr':[]
    }


    pbar = None
    if USE_TQDM:
        # Use progress bar
        pbar = tqdm(total=train_args.iterations, desc="Training", smoothing=.5)

    summarizer = None
    if USE_TENSORBOARD:
        raise NotImplementedError("Tensorboard not supported yet")
        summarizer = SummaryWriter(
            ...
        )

    # Set up epoch
    loss_accum=0.
    loss_norm=0

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

            if iter <= 15_000:
                # Densify
                model.update_densification_stats(
                    xys=rendering_pkg.xys,
                    radii=rendering_pkg.radii,
                    visibility_mask=rendering_pkg.visibility_mask
                )
                if iter >= 1000 and iter%500 == 0:
                    model.densify(
                        grad_threshold=2e-8,
                        max_scale=0.01 * dataset.scene_extend,
                        min_opacity=0.005,
                    )

                if iter%3000 == 0:
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



        # End iter
        scheduler.step()
        camera.to('cpu')
        torch.cuda.empty_cache()

    pbar.close()

    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")

    mp = ModelParams(parser)
    dp = DataParams(parser)
    tp = TrainParams(parser)
    pp = PipeLineParams(parser)

    model_args = mp.extract()
