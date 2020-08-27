import csv
import numpy as np
import cv2
import math
import os

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, init_logger_ipol, remove_dataparallel_wrapper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def normalize(data, benchmark):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data / benchmark)


def variable_to_cv2_image(varim, benchmark):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * benchmark).clip(0, benchmark)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def Normalization(data, sigma):
    data = data.astype(np.float64)
    sigma = sigma.astype(np.float64)
    data_max = np.max(data)
    data2 = (data / data_max) * 255.
    sigma = (sigma / data_max) * 255.
    return data2, sigma, data_max

def random_sort(data, sigma):
    data_shape = data.shape
    data = data.flatten()

    weight = np.exp(data / sigma)
    prob_dist = weight / np.sum(weight)
    sorted_data = np.random.choice(data, np.size(data), p = prob_dist, replace = False)
    sorted_data = sorted_data.reshape(data_shape)
    return sorted_data

def ADD_Noise(data, sigma):
    global n
    noisy = np.random.normal(0, sigma, (n, n))
    return data + noisy


def deal(imorig, data_upper):
    imorig = np.array(imorig, dtype="float64")
    imorig = imorig * data_upper / 255.
    original_img = imorig
    original_img[original_img == 0] = 1
    return original_img


def Error_Analyes(original_img, noisy_img, output_img):
    error_out = np.array(np.fabs(output_img - original_img))
    error_noisy = np.array(np.fabs(noisy_img - original_img))

    MAE_output = np.sum(error_out) / (np.size(error_out))
    MAE_noisy = np.sum(error_noisy) / (np.size(error_noisy))

    MSE_output = np.sum(error_out ** 2) / (np.size(error_out))
    MSE_noisy = np.sum(error_noisy ** 2) / (np.size(error_noisy))

    MRE_output = np.sum(error_out / original_img) / (np.size(error_out))
    MRE_noisy = np.sum(error_noisy / original_img) / (np.size(error_noisy))

    return MAE_output, MAE_noisy, MSE_output, MSE_noisy, MRE_output, MRE_noisy


def test_ffdnet(imorig, sigma, data_max, eps, datasets):
    r"""Denoises an input image with FFDNet
    """
    args = {
        'cuda': False,
        'add_noise': True,
        'noise_sigma': sigma / data_max
    }

    logger = init_logger_ipol()

    try:
        rgb_den = False
    except:
        raise Exception('Could not open the input image')

    if rgb_den:
        in_ch = 3
        model_fn = 'models/net_rgb.pth'
        imorig = cv2.imread(args['input'])
        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)

    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3] % 2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

    imorig = normalize(imorig, data_max)
    imorig = torch.Tensor(imorig)

    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)

    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    model.eval()

    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if args['add_noise']:
        noise = torch.FloatTensor(imorig.size()).normal_(mean=0, std=args['noise_sigma'])
        imnoisy_o = imorig + noise
        imnoisy = random_sort(imnoisy_o,args['noise_sigma'])
    else:
        imnoisy = imorig.clone()

    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([args['noise_sigma']]).type(dtype))

    start_t = time.time()

    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
    stop_t = time.time()

    if expanded_h:
        imorig = imorig[:, :, :-1, :]
        outim = outim[:, :, :-1, :]
        imnoisy = imnoisy[:, :, :-1, :]

    if expanded_w:
        imorig = imorig[:, :, :, :-1]
        outim = outim[:, :, :, :-1]
        imnoisy = imnoisy[:, :, :, :-1]

    if rgb_den:
        logger.info("### RGB denoising ###")
    else:
        logger.info("### Grayscale denoising ###")
    if args['add_noise']:
        psnr = batch_psnr(outim, imorig, 1.)
        psnr_noisy = batch_psnr(imnoisy, imorig, 1.)

        logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
        logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
    else:
        logger.info("\tNo noise was added, cannot compute PSNR")
    logger.info("\tRuntime {0:0.4f}s".format(stop_t - start_t))


    noisyimg = variable_to_cv2_image(imnoisy, data_max)
    outimg = variable_to_cv2_image(outim, data_max)
    noisyimg_name = datasets + '_' + 'noisy' + '_' + str(eps) + '.png'
    output_name = datasets + '_' + 'output' + '_' + str(eps) + '.png'
    cv2.imwrite(noisyimg_name, noisyimg)
    cv2.imwrite(output_name, outimg)

    return noisyimg, outimg


def main(eps, column, datasets):
    data_max = np.max(column)

    if datasets == 'dataset_name_1':
        O_sigma = ((2 * math.log(1.25 / (3.5 * 10 ** (-5)))) ** 0.5) * 196. / eps
    else:
        O_sigma = ((2 * math.log(1.25 * column.size)) ** 0.5) * (1.0-0.) / eps

    print("Original_sigma: %f \n" % O_sigma)

    noisy_img, output_img = test_ffdnet(column, O_sigma, data_max, eps, datasets)

    MAE_output, MAE_noisy, MSE_output, MSE_noisy, MRE_output, MRE_noisy = Error_Analyes(column, noisy_img, output_img)

    print('MAE', MAE_noisy, MAE_output)
    print('MSE', MSE_noisy, MSE_output)
    print('MRE', MRE_noisy, MRE_output)
    print('-----------------------------------------------------\n')
    with open("Results.csv", "a") as result_csv:
        writer = csv.writer(result_csv)
        data_row = [eps, O_sigma, MAE_noisy, MAE_output, MSE_noisy, MSE_output, MRE_noisy, MRE_output]
        writer.writerow(data_row)


if __name__ == '__main__':
    
    column = data
    n = data.shape

    with open("Results.csv", "a") as result_csv:
        writer = csv.writer(result_csv)
        writer.writerow([datasets])
        data_row = ["n", n]
        writer.writerow(data_row)
        csv_head = ["eps", "Original_sigma", "MAE_noisy", "MAE_filter", "MSE_noisy",
                    "MSE_filter", "MRE_noisy",
                    "MRE_filter"]
        writer.writerow(csv_head)

    for eps in np.arange(0.1, 1.1, 0.1):
        print(eps)
        main(eps, column, datasets)
