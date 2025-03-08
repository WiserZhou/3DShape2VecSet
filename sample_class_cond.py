import argparse
import numpy as np
import mcubes
import torch
import trimesh
import models_class_cond, models_ae
from pathlib import Path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True)  # Autoencoder model name
    parser.add_argument('--ae-pth', type=str, required=True)  # Path to autoencoder checkpoint
    parser.add_argument('--dm', type=str, required=True)  # Diffusion model name
    parser.add_argument('--dm-pth', type=str, required=True)  # Path to diffusion model checkpoint
    args = parser.parse_args()
    print(args)

    # Create output directory
    Path("class_cond_obj/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)

    # Set device to CUDA
    device = torch.device('cuda:0')

    # Load and prepare autoencoder model
    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    # Load and prepare diffusion model
    model = models_class_cond.__dict__[args.dm]()
    model.eval()
    model.load_state_dict(torch.load(args.dm_pth)['model'])
    model.to(device)

    # Set up 3D grid for sampling
    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    # Set sampling parameters
    total = 1000
    iters = 100

    with torch.no_grad():
        for category_id in [18]:  # Loop over categories (only category 18 in this case)
            print(category_id)
            for i in range(1000//iters):  # Generate samples in batches
                # Sample from the diffusion model
                sampled_array = model.sample(cond=torch.Tensor([category_id]*iters).long().to(device), batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()

                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                for j in range(sampled_array.shape[0]):  # Process each sample in the batch
                    # Decode the sample using the autoencoder
                    logits = ae.decode(sampled_array[j:j+1], grid)
                    logits = logits.detach()
                    
                    # Convert logits to 3D volume
                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    
                    # Apply marching cubes to extract mesh
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    # Adjust vertex positions
                    verts *= gap
                    verts -= 1

                    # Create and export mesh
                    m = trimesh.Trimesh(verts, faces)
                    m.export('class_cond_obj/{}/{:02d}-{:05d}.obj'.format(args.dm, category_id, i*iters+j))