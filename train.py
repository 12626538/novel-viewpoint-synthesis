import argparse
import os
import numpy as np

try:
    from tqdm import tqdm
    USE_TQDM=True
except ModuleNotFoundError:
    USE_TQDM=False

import torch
from torchvision.utils import save_image
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr


try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD=True
except ModuleNotFoundError:
    print("Warning! Tensorboard not available")
    USE_TENSORBOARD=False

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet
from src.utils.loss_utils import L1Loss,DSSIMLoss,SSIM,CombinedLoss
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams,get_args
from src.camera import OptimizableCamera

def train_report(
    iter:int, # !first iter is 1, not 0!
    model:Gaussians,
    dataset:DataSet,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams,
    rendering_pkg:RenderPackage,
    smoothed_loss:float,
    loss_fn:callable,
    pbar:'tqdm'=None,
    summarizer:'SummaryWriter'=None,
):
    # Update progress bar
    if pbar is not None and iter%10 == 0:
        pbar.set_postfix({
            'loss':f"{smoothed_loss:.2e}",
            '#splats': f"{model.num_points:.2e}",
        }, refresh=False)
        pbar.update(10)

    # Update summarizer
    if summarizer is not None:
        if iter%10 == 0:
            summarizer.add_scalar("Train/Loss", smoothed_loss, iter)
            for group in model.optimizer.param_groups:
                summarizer.add_scalar(f"LearningRate/{group['name']}", group['lr'], iter)

            summarizer.add_scalar("Pruning/Num of splats", model.num_points, iter)
            if iter <= train_args.densify_until+10:
                summarizer.add_scalar("Pruning/Num splats split",model._n_split, iter)
                summarizer.add_scalar("Pruning/Num splats cloned",model._n_clone, iter)
                summarizer.add_scalar("Pruning/Num splats pruned",model._n_prune, iter)

            # summarizer.add_histogram("dev/radii", rendering_pkg.radii[rendering_pkg.visibility_mask].detach().cpu().numpy().clip(0,train_args.max_screen_size+1), iter)

        if iter==1 or iter%100 == 0:

            camera = dataset.cameras[0]
            render = model.render(camera, bg=torch.zeros(3,device=device)).rendering
            save_image(render,"renders/latest.png")

            if iter==1 or iter%300 == 0:
                summarizer.add_image(
                    "Dev/Render",
                    render,
                    iter
                )

    # Save image
    if iter in train_args.save_at:

        path = os.path.join(pipeline_args.model_dir, f'iter_{iter}')
        os.makedirs(path, exist_ok=True)
        model.to_ply(path=path)

        print("Model saved to",path)

    # Test model
    if iter in train_args.test_at:
        torch.cuda.empty_cache()

        # Compute average over PSNR and loss
        metrics = {
            'PSNR':psnr,
            'loss':loss_fn,
            'SSIM':SSIM(device=device)
            # TODO: add LPIPS
        }
        stats_lsts = {
            metric:[] for metric in metrics
        }
        reduction = {
            'PSNR': np.mean,
            'loss': np.mean,
            'SSIM': np.mean,
        }

        print("TESTING", "+"*50)

        if not pipeline_args.random_background and pipeline_args.white_background:
            bg = torch.ones(3).float().to(device)
        else:
            bg = torch.zeros(3).float().to(device)

        # Get testing cameras
        dataset.test()
        for cam in dataset:

            pkg = model.render(cam, bg=bg)

            # Get result for each metric to test on
            for metric,metric_func in metrics.items():

                # Compute metric
                stat = metric_func(pkg.rendering, cam.gt_image)

                # convert Tensor with a single number to a float
                if isinstance(stat, torch.Tensor) and stat.squeeze().ndim == 0:
                    stat = stat.item()

                # Add metric
                stats_lsts[metric].append( stat )

            # Save rendered images in summarizer
            if summarizer is not None:
                summarizer.add_images(f"test/{cam.name}", torch.stack((pkg.rendering, cam.gt_image)), iter )

            # TODO: save images during testing
            # save_image(torch.hstack((pkg.rendering, cam.gt_image)),cam.name)

        # Reduce results
        stats = {}
        for metric,values in stats_lsts.items():
            # Compute the final value
            stats[metric] = reduction[metric](values)

            print(metric, stats[metric])
            if summarizer is not None:
                summarizer.add_scalar(f"test/{metric}", stats[metric], iter)

        dataset.train()

def train_loop(
    model:Gaussians,
    dataset:DataSet,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams
):

    # Set up loss
    loss_fn = CombinedLoss(
        ( L1Loss(), 1.-train_args.lambda_dssim ),
        ( DSSIMLoss(device=device), train_args.lambda_dssim )
    )

    # Keep a running (smoothed) loss
    loss_smooth_mult = 1-1/len(dataset)
    loss_smooth = 0.

    # Use Tensorboard
    summarizer = None
    if USE_TENSORBOARD:
        print("Tensorboard running at", pipeline_args.log_dir)
        summarizer = SummaryWriter(
            log_dir=pipeline_args.log_dir
        )
    # Set background
    background = None
    if not pipeline_args.random_background:
        background = torch.tensor([1,1,1] if pipeline_args.white_background else [0,0,0], dtype=torch.float32, device=device)

    # Use TQDM progress bar
    pbar = None
    if USE_TQDM:
        pbar = tqdm(total=train_args.iterations, desc="Training", smoothing=.5)

    cam_optim = torch.optim.Adam(
        params=[
            {'name':cam.name, 'params':cam.parameters()}
            for cam in dataset.cameras if isinstance(cam, OptimizableCamera)
        ]
    )
    def lr_func(step):
        decay_steps = 2500
        lr_mult_init = 1e-8
        lr_mult_final = 1
        if step <= decay_steps:
            # starts at 1, goes to 0
            cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))

            return cosine_decay * lr_mult_init + (1-cosine_decay)*lr_mult_final
        else:
            return 1
    cam_schedule = torch.optim.lr_scheduler.LambdaLR(cam_optim, lr_lambda=lr_func)

    dataset_cycle = dataset.cycle()
    for iter in range(1,train_args.iterations+1):

        model.optimizer.zero_grad(set_to_none=True)
        cam_optim.zero_grad()

        # Forward pass
        camera = next(dataset_cycle)
        rendering_pkg = model.render(camera, bg=background)
        loss = loss_fn(rendering_pkg.rendering, camera.gt_image)

        # Compute gradients
        loss.backward()

        # Keep a smoothed loss
        if loss_smooth == 0:
            loss_smooth = loss.item()
        else:
            loss_smooth = loss_smooth_mult * loss_smooth \
                + (1-loss_smooth_mult) * loss.item()

        # Optimize parameters
        model.optimizer.step()
        cam_optim.step()

        with torch.no_grad():

            # Densify
            if iter < train_args.densify_until:

                model.update_densification_stats(
                    xys=rendering_pkg.xys,
                    radii=rendering_pkg.radii,
                    visibility_mask=rendering_pkg.visibility_mask
                )

                if train_args.densify_from < iter and iter % train_args.densify_every == 0:
                    model.densify(
                        grad_threshold=train_args.grad_threshold,
                        max_density=train_args.max_density * dataset.scene_extend,
                        min_opacity=train_args.min_opacity,
                        max_world_size=0.1*dataset.scene_extend,
                        # TODO see what works here
                        max_screen_size=train_args.max_screen_size if iter > train_args.reset_opacity_from else None,
                    )

            # Reset opacity
            if train_args.reset_opacity_from <= iter <= train_args.reset_opacity_until \
            and iter % train_args.reset_opacity_every == 0:
                model.reset_opacity()

            # Increase SH degree used
            if iter%train_args.oneup_sh_every == 0:
                model.oneup_sh_degree()

            # Report on iter
            train_report(
                iter=iter,
                model=model,
                dataset=dataset,
                device=device,
                train_args=train_args,
                pipeline_args=pipeline_args,
                rendering_pkg=rendering_pkg,
                smoothed_loss=loss_smooth,
                loss_fn=loss_fn,
                pbar=pbar,
                summarizer=summarizer
            )

        cam_schedule.step()

    if pbar is not None:
        pbar.close()

if __name__ == '__main__':

    args,(data_args,model_args,train_args,pipeline_args) = get_args(
        DataParams, ModelParams, TrainParams, PipeLineParams
    )

    # parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
    # dp = DataParams(parser)
    # mp = ModelParams(parser)
    # tp = TrainParams(parser)
    # pp = PipeLineParams(parser)

    # args = parser.parse_args()

    # data_args = dp.extract(args)
    # model_args = mp.extract(args)
    # train_args = tp.extract(args)
    # pipeline_args = pp.extract(args)

    if pipeline_args.no_pbar:
        USE_TQDM = False

    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    # 3DU datasets are from intrinsics and extrinsics
    if "3du_data_" in data_args.source_path or "jesse" in data_args.source_path:
        print("Reading dataset from intrinsics/extrinsics...")
        dataset = DataSet.from_intr_extr(
            device=args.device,
            **vars(data_args)
        )

    # Paper datasets are COLMAP formatted
    else:
        print("Reading dataset from COLMAP...")
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    # Use checkpoint
    if pipeline_args.load_checkpoint:
        print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
        model = Gaussians.from_ply(
            device=args.device,
            path_to_ply_file=pipeline_args.load_checkpoint,
            **vars(model_args)
        )

    # Initialize 3DU data randomly
    elif args.source_path[:-1].endswith("3du_data_"):
        print("Initializing model randomly with {} points in an area of size {:.1f}".format(
            model_args.num_points,
            dataset.scene_extend
        ))
        model = Gaussians(
            device=args.device,
            scene_extend=dataset.scene_extend,
            **vars(model_args),
        )

    # Initialize from COLMAP `points3D.txt` file
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
