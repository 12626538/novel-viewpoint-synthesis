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
from src.utils.metric_utils import get_loss_fn,SSIM,LPIPS
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams,get_args

def train_report(
    iter:int, # !first iter is 1, not 0!
    model:Gaussians,
    dataset:DataSet,
    rendering_pkg:RenderPackage,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams,
    loss_pkg:dict[str,float],
    smoothed_loss:float,
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
            for name in loss_pkg:
                summarizer.add_scalar(f"Train/Loss - {name}", loss_pkg[name], iter)

            for group in model.optimizer.param_groups:
                summarizer.add_scalar(f"LearningRate/{group['name']}", group['lr'], iter)

            summarizer.add_scalar("Pruning/Num of splats", model.num_points, iter)
            if iter <= train_args.densify_until+10:
                summarizer.add_scalar("Pruning/Num splats split",model._n_split, iter)
                summarizer.add_scalar("Pruning/Num splats cloned",model._n_clone, iter)
                summarizer.add_scalar("Pruning/Num splats pruned (total)",model._n_prune, iter)
                summarizer.add_scalar("Pruning/Num splats pruned by opacity",model._n_prune_opacity, iter)
                summarizer.add_scalar("Pruning/Num splats pruned by view radius",model._n_prune_radii, iter)
                summarizer.add_scalar("Pruning/Num splats pruned by global radius",model._n_prune_scale, iter)

            if rendering_pkg.blur_quat is not None:
                summarizer.add_scalar("Blurring/average weight quat", rendering_pkg.blur_quat[rendering_pkg.visibility_mask].mean(), iter)
            if rendering_pkg.blur_scale is not None:
                summarizer.add_scalar("Blurring/average weight scale", rendering_pkg.blur_scale[rendering_pkg.visibility_mask].mean(), iter)

    if iter==1 or iter%100 == 0:

        camera = dataset.cameras[0]
        render = model.render(camera, bg=torch.zeros(3,device=device), blur=False).rendering
        save_image(render,f"renders/latest_{pipeline_args.model_name}.png")

    # Save model
    if iter in train_args.save_at:

        path = os.path.join(pipeline_args.model_dir, f'iter_{iter}')
        os.makedirs(path, exist_ok=True)
        model.to_ply(path=path)

        print("Model saved to",path)

    # Test model
    if iter in train_args.test_at:
        torch.cuda.empty_cache()

        print("EVALUATING", f"iter={iter}", "+"*50)

        if not pipeline_args.random_background and pipeline_args.white_background:
            bg = torch.ones(3).float().to(device)
        else:
            bg = torch.zeros(3).float().to(device)

        def eval_set(partition:str, eval_every:int=1):
            """
            Run evaluations on partition
            either "train" or "test"

            Use `eval_every` to test a subset of the data
            (ie `eval_every=2` will test every second sample)
            default is every 1 (ie all).
            """
            print("Set:",partition.upper())

            # Metrics to run
            metric_funcs = {
                'PSNR':psnr,
                'SSIM':SSIM(device=device),
                'LPIPS': LPIPS(net='alex'),
            }
            metric_reduction = {
                'PSNR': np.mean,
                'SSIM': np.mean,
                'LPIPS': np.mean,
            }
            stats_lsts = {
                metric:[] for metric in metric_funcs
            }

            # On last iter, also save to disk
            out_dir = None
            if iter == train_args.iterations:
                out_dir = os.path.join(pipeline_args.log_dir, f"iter_{iter}", "renders", partition)
                os.makedirs(out_dir, exist_ok=True)

            # Get testing cameras
            for i,cam in enumerate(dataset.iter(partition)):

                # Only evaluate every so often
                if i%eval_every != 0:
                    continue

                pkg = model.render(cam, bg=bg, blur=False)

                # Get result for each metric to test on
                for metric,metric_func in metric_funcs.items():

                    # Compute metric
                    stat = metric_func(pkg.rendering, cam.gt_image)

                    # convert Tensor with a single number to a float
                    if isinstance(stat, torch.Tensor) and stat.squeeze().ndim == 0:
                        stat = stat.item()

                    # Add metric
                    stats_lsts[metric].append( stat )

                # Save rendered images
                if summarizer is not None:
                    summarizer.add_images(f"{partition}_renders/{cam.name}", torch.stack((pkg.rendering, cam.gt_image)), iter )

                if out_dir is not None:
                    save_image(
                        pkg.rendering,
                        os.path.join(out_dir, cam.name)
                    )

            # Reduce results
            stats = {}
            for metric,values in stats_lsts.items():
                # Compute the final value
                stats[metric] = metric_reduction[metric](values)

                print(metric, stats[metric])
                if summarizer is not None:
                    summarizer.add_scalar(f"{partition}_metric/{metric}", stats[metric], iter)

        eval_set("train", eval_every=10)
        eval_set("test")

def train_loop(
    model:Gaussians,
    dataset:DataSet,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams
):

    # initialize lr schedule
    model.init_lr_schedule(
        warmup_until=len(dataset),
        decay_for=train_args.iterations//3,
        decay_from=train_args.iterations - train_args.iterations//3
    )

    # Set up loss
    loss_fn = get_loss_fn(vars(train_args))

    # Keep a running (smoothed) loss for logging
    loss_smooth_mult = .9
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

    dataset_cycle = dataset.iter("train",cycle=True,shuffle=True)
    for iter in range(1,train_args.iterations+1):

        model.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        camera = next(dataset_cycle)
        rendering_pkg = model.render(camera, bg=background, blur=pipeline_args.do_blur)

        loss,loss_pkg = loss_fn(rendering_pkg.rendering, camera.gt_image)

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

        with torch.no_grad():

            # Densify
            if iter < train_args.densify_until:

                model.update_densification_stats(
                    xys=rendering_pkg.xys,
                    radii=rendering_pkg.radii,
                    visibility_mask=rendering_pkg.visibility_mask
                )

                if train_args.densify_from < iter and iter % train_args.densify_every == 0:
                    screen_size = max(camera.H, camera.W)
                    model.densify(
                        grad_threshold=train_args.grad_threshold / (.5*screen_size),
                        max_density=train_args.max_density,
                        min_opacity=train_args.min_opacity,
                        max_world_size=0.1*dataset.scene_extend,
                        max_screen_size=train_args.max_screen_size*screen_size if iter > train_args.reset_opacity_from else None,
                    )

            # Reset opacity
            if train_args.reset_opacity_from <= iter <= train_args.reset_opacity_until \
            and iter % train_args.reset_opacity_every == 0:
                model.reset_opacity(value=train_args.min_opacity*2.0)

            # Increase SH degree used
            if iter%train_args.oneup_sh_every == 0:
                model.oneup_sh_degree()

            # Report on iter
            train_report(
                iter=iter,
                model=model,
                dataset=dataset,
                rendering_pkg=rendering_pkg,
                device=device,
                train_args=train_args,
                pipeline_args=pipeline_args,
                smoothed_loss=loss_smooth,
                loss_pkg=loss_pkg,
                pbar=pbar,
                summarizer=summarizer
            )

        if model.lr_schedule is not None:
            model.lr_schedule.step()

    if pbar is not None:
        pbar.close()

if __name__ == '__main__':

    # Get args
    args,(data_args,model_args,train_args,pipeline_args) = get_args(
        DataParams, ModelParams, TrainParams, PipeLineParams,
        save_args=True
    )

    if pipeline_args.no_pbar:
        USE_TQDM = False

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set torch device
    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    # 3DU datasets from src/preprocessing/convert_3du.py
    if "3du_data_" in data_args.source_path:
        print("Reading from 3DU dataset...")
        dataset = DataSet.from_3du(
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

    # Initialize 3DU pointcloud
    elif "3du_data" in data_args.source_path:
        fname = os.path.join(data_args.source_path, "point_cloud.ply")

        print(f"Loading .ply model from {fname}")
        model = Gaussians.from_ply(
            path_to_ply_file=fname,
            device=args.device,
            **vars(model_args)
        )
    elif os.path.isdir(os.path.join(data_args.source_path,"sparse")):
        fname = os.path.join(data_args.source_path, "sparse","0","points3D.txt")
        print(f"Loading COLMAP model from {fname}")
        model = Gaussians.from_colmap(
            device=args.device,
            path_to_points3D_file=fname,
            **vars(model_args)
        )
    else:
        raise ValueError("Specify initialization method!")
        # Random initialization
        print("Initializing model randomly with {} points in a bounding box of size {:.1f}".format(
            model_args.num_points,
            dataset.scene_extend
        ))
        model = Gaussians(
            device=args.device,
            scene_extend=dataset.scene_extend,
            **vars(model_args),
        )

        # Initialize from COLMAP `points3D.txt` file
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
