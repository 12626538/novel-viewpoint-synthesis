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
        if iter%10 == 0:
            summarizer.add_scalar("Loss/Train", loss, iter)
            for group in model.optimizer.param_groups:
                summarizer.add_scalar(f"LearningRate/{group['name']}", group['lr'], iter)
            summarizer.add_scalar("Model/NumSplats", model.num_points, iter)

            summarizer.add_scalar("Pruning/n_split",model._n_split,iter)
            summarizer.add_scalar("Pruning/n_clone",model._n_clone,iter)
            summarizer.add_scalar("Pruning/n_prune",model._n_prune,iter)

        if (iter-1)%100 == 0:
            grads = model._xys_grad_accum / model._xys_grad_norm
            grads[grads.isnan()] = 0.
            grads2=grads.detach().cpu().numpy()
            summarizer.add_histogram(
                "Pruning/2DGrads",
                grads2,
                iter
            )

            camera = dataset.cameras[0]
            render = model.render(camera, bg=torch.zeros(3,device=device)).rendering.permute(2,0,1)
            save_image(render,"renders/latest.png")
            summarizer.add_image(
                "Dev/Render",
                render,
                iter
            )

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
        summarizer = SummaryWriter(
            log_dir=pipeline_args.log_dir
        )

    dataset_cycle = dataset.cycle()

    for iter in range(1,train_args.iterations+1):

        camera = next(dataset_cycle)
        rendering_pkg = model.render(camera)
        loss = loss_fn(rendering_pkg.rendering, camera.gt_image)
        loss.backward()

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
                        max_screen_size=20 if iter > train_args.reset_opacity_from else None
                    )

            # Reset opacity
            if train_args.reset_opacity_from <= iter <= train_args.reset_opacity_until \
            and iter % train_args.reset_opacity_every == 0:
                model.reset_opacity()


            # TODO: move this to some method in Gaussians
            t = iter / train_args.iterations
            lr = np.exp( (1-t)*np.log(0.00016 * dataset.scene_extend) + t*np.log(0.0000016* dataset.scene_extend) )

            for group in model.optimizer.param_groups:
                if group['name'] == 'means':
                    group['lr'] = lr
                    break
            # </TODO>

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
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

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
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    dataset = ColmapDataSet(device=args.device, **vars(data_args))

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

    dataset.cameras = dataset.cameras[:100]

    train_loop(
        model=model,
        dataset=dataset,
        device=args.device,
        train_args=train_args,
        pipeline_args=pipeline_args
    )
