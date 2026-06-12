import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import RetinexDuelSambaFusionFinalization

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr

from utils import calculate_ssim
from utils import calculate_psnr
import time

import torch.nn.functional as F

def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

    
def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/mnt/data0/NTIRE/NTIRENoiseLLE/Validation/jdllie_in',
                        help='Input image or folder')

    parser.add_argument('-g', '--gt', type=str,
                        default=None,
                        help='groundtruth image folder (optional)')
    parser.add_argument('-w', '--weight', type=str,
                        default='/mnt/data0/NTIREexperiemnts/experiments/RetinexDualNTIRENoiseLLIE/models/net_g_14000.pth',
                        help='path for model weights')

    parser.add_argument('-o', '--output', type=str, default='results/RetinexDual/NTIREJNLLIE', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=5000,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhance_weight_path = args.weight

    EnhanceNet = RetinexDuelSambaFusionFinalization(
                                                    in_channels= 3,
                                                    out_channels= 3,
                                                    L_n_feat=16,
                                                    R_n_feat=16).to(device)

    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
    EnhanceNet.eval()
    print_network(EnhanceNet)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    total_inference_time = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path) 
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt 
        file_name = path.split('/')[-1]

        # Load ground truth only if path is provided
        gt_img = None
        if gt_path and os.path.exists(os.path.join(gt_path, file_name)):
            gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        img_tensor = img2tensor(img).to(device) / 255. 
        img_tensor = img_tensor.unsqueeze(0) 
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())
        img_tensor = check_image_size(img_tensor) 
        start_time = time.time()
        with torch.no_grad():
            output = EnhanceNet(img_tensor) 
        output = output
        end_time = time.time()
        inference_time = end_time - start_time

        # Accumulate inference time
        total_inference_time += inference_time

        output = output[0][:, :, :h, :w]
        output_img = tensor2img(output) 
        gray = True
        
        # Calculate metrics only if ground truth is available
        if gt_img is not None:
            print('output_img.shape', output_img.shape, "input_img.shape", gt_img.shape)
            ssim = calculate_ssim(output_img, gt_img) 
            psnr = calculate_psnr(output_img, gt_img) 
            lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                                2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1) 
            ssim_all += ssim
            psnr_all += psnr
            lpips_all += lpips_value
            num_img += 1
            print('num_img', num_img)
            print('ssim', ssim)
            print('psnr', psnr)
            print('lpips_value', lpips_value)
        else:
            print('output_img.shape', output_img.shape)
            num_img += 1
            print('num_img', num_img)
        print('inference_time', inference_time)
        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path) 

        pbar.update(1)
    pbar.close()
    
    # Print metrics only if ground truth was available
    if args.gt:
        print('avg_ssim:%f' % (ssim_all / num_img))
        print('avg_psnr:%f' % (psnr_all / num_img))
        print('avg_lpips:%f' % (lpips_all / num_img))
    print('avg_inference_time: %f seconds' % (total_inference_time / num_img))


if __name__ == '__main__':
    main()
