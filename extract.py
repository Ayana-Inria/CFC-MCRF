# -*- coding: utf-8 -*-
"""
Extraction utilities for saving posteriors and activations for the MATLAB fusion code.

This script saves the tensors using the filenames and variable names expected by
multires_CFCCRF.m:

    <dataset>_<CNN_model>_post6_PAN.mat   -> post_pan
    <dataset>_<CNN_model>_post6_HYS.mat   -> post_hys
    <dataset>_<CNN_model>_act_8_PAN.mat   -> act_pan
    <dataset>_<CNN_model>_act_8_HYS.mat   -> act_hys
"""

import os
import numpy as np
import torch
import scipy.io as sio

from utils.utils import *
from utils.utils_dataset import *

try:
    from osgeo import gdal
except ImportError:
    gdal = None


PRISMA_TENSORS_FOLDER = 'PRISMA_Tensors'


def _read_prisma_tile(id_, folder=input_folder):
    if gdal is None:
        raise ImportError('GDAL is required to read PRISMA .tif files. Install it with conda install -c conda-forge gdal.')

    pan_path = os.path.join(folder, id_ + '_Cube.tif')
    hys_path = os.path.join(folder, id_ + '_VNIR_SWIR.tif')

    pan = gdal.Open(pan_path, gdal.GA_ReadOnly).ReadAsArray()
    hys = gdal.Open(hys_path, gdal.GA_ReadOnly).ReadAsArray()

    pan = np.asarray(np.expand_dims(pan, axis=0), dtype='float32')

    # Same band selection used in the dataset loader.
    hys1 = np.asarray(hys, dtype='float32')[3:66, :, :]   # bands with artifacts removal
    hys2 = np.asarray(hys, dtype='float32')[69:, :, :]
    hys = np.concatenate((hys1, hys2), 0)

    pan[np.isnan(pan)] = 0
    hys[np.isnan(hys)] = 0

    pan = 1 / 255 * pan
    hys = 1 / 255 * hys

    return pan, hys


def _save_mat(path, variable_name, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.savemat(path, {variable_name: array})
    print('Saved {}'.format(path))


def _split_output(output):
    """Handle networks returning either (logits, activations) or (pan_logits, hys_logits, activations)."""
    logits = output[0]
    hys_logits = None
    activations = None

    if len(output) == 2:
        activations = output[1]
    elif len(output) >= 3:
        hys_logits = output[1]
        activations = output[2]

    return logits, hys_logits, activations


def _get_activation(activations, activation_index=-1):
    """Select the activation map used as act_8 by the MATLAB routine."""
    if activations is None:
        return None
    if isinstance(activations, dict):
        values = list(activations.values())
    else:
        values = list(activations)
    if len(values) == 0:
        return None
    return values[activation_index]


def _to_matlab_image_layout(array):
    """Convert C x H x W tensors to H x W x C for MATLAB."""
    if array.ndim == 3:
        return np.transpose(array, (1, 2, 0))
    return array


def extract_tensors(net, ids, dataset, CNN_model, folder=input_folder,
                    save_folder=PRISMA_TENSORS_FOLDER, window_size=WINDOW_SIZE,
                    step=None, activation_index=-1):
    """
    Extract PAN/HYS posteriors and act_8 activations, then save .mat files.

    Parameters
    ----------
    net : torch.nn.Module
        Trained network.
    ids : list[str]
        PRISMA tile ids to process. If more than one id is given, outputs are
        saved with the id appended to the dataset name to avoid overwriting.
    dataset : str
        Dataset name used in MATLAB filenames.
    CNN_model : str
        Model name used in MATLAB filenames.
    folder : str
        Folder containing *_Cube.tif and *_VNIR_SWIR.tif files.
    save_folder : str
        Folder where MATLAB tensors are saved. Default: PRISMA_Tensors.
    window_size : tuple[int, int]
        PAN patch size.
    step : int or None
        Sliding-window step. If None, uses non-overlapping windows.
    activation_index : int
        Which activation returned by the network is saved as act_8.
        Default -1 uses the last activation.
    """
    if step is None:
        step = window_size[0]

    device = next(net.parameters()).device
    net.eval()

    with torch.no_grad():
        for id_ in ids:
            pan, hys = _read_prisma_tile(id_, folder=folder)

            post_pan = None
            count_pan = None
            post_hys = None
            count_hys = None
            act_pan = None
            count_act_pan = None
            act_hys = None
            count_act_hys = None

            for x, y, w, h in sliding_window(pan[0], step=step, window_size=window_size):
                scale = 3
                xh, yh = x // scale, y // scale
                wh, hh = max(1, w // scale), max(1, h // scale)

                pan_patch = pan[:, x:x + w, y:y + h]
                hys_patch = hys[:, xh:xh + wh, yh:yh + hh]

                pan_patch = torch.from_numpy(pan_patch[None]).float().to(device)
                hys_patch = torch.from_numpy(hys_patch[None]).float().to(device)

                output = net(pan_patch, hys_patch)
                logits, hys_logits, activations = _split_output(output)

                # The original training uses CrossEntropy2d, so logits may be raw scores.
                # Softmax is safer than exp unless the network already returns log-softmax.
                pan_prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

                if post_pan is None:
                    post_pan = np.zeros((pan_prob.shape[0], pan.shape[1], pan.shape[2]), dtype='float32')
                    count_pan = np.zeros((1, pan.shape[1], pan.shape[2]), dtype='float32')

                post_pan[:, x:x + w, y:y + h] += pan_prob
                count_pan[:, x:x + w, y:y + h] += 1

                if hys_logits is not None:
                    hys_prob = torch.softmax(hys_logits, dim=1).cpu().numpy()[0]

                    if post_hys is None:
                        post_hys = np.zeros((hys_prob.shape[0], hys.shape[1], hys.shape[2]), dtype='float32')
                        count_hys = np.zeros((1, hys.shape[1], hys.shape[2]), dtype='float32')

                    post_hys[:, xh:xh + hys_prob.shape[1], yh:yh + hys_prob.shape[2]] += hys_prob
                    count_hys[:, xh:xh + hys_prob.shape[1], yh:yh + hys_prob.shape[2]] += 1

                activation = _get_activation(activations, activation_index=activation_index)
                if activation is not None:
                    activation = activation.cpu().numpy()[0]

                    # If the selected activation has PAN resolution, save it as act_pan.
                    # If it has HYS resolution, save it as act_hys.
                    if activation.shape[1] == pan_prob.shape[1] and activation.shape[2] == pan_prob.shape[2]:
                        if act_pan is None:
                            act_pan = np.zeros((activation.shape[0], pan.shape[1], pan.shape[2]), dtype='float32')
                            count_act_pan = np.zeros((1, pan.shape[1], pan.shape[2]), dtype='float32')
                        act_pan[:, x:x + activation.shape[1], y:y + activation.shape[2]] += activation
                        count_act_pan[:, x:x + activation.shape[1], y:y + activation.shape[2]] += 1
                    else:
                        if act_hys is None:
                            act_hys = np.zeros((activation.shape[0], hys.shape[1], hys.shape[2]), dtype='float32')
                            count_act_hys = np.zeros((1, hys.shape[1], hys.shape[2]), dtype='float32')
                        act_hys[:, xh:xh + activation.shape[1], yh:yh + activation.shape[2]] += activation
                        count_act_hys[:, xh:xh + activation.shape[1], yh:yh + activation.shape[2]] += 1

            count_pan[count_pan == 0] = 1
            post_pan = post_pan / count_pan

            # If the network does not produce a separate HYS posterior, use the PAN posterior
            # downsampled to HYS grid so MATLAB still receives post_hys.
            if post_hys is None:
                post_hys = post_pan[:, ::3, ::3]
            else:
                count_hys[count_hys == 0] = 1
                post_hys = post_hys / count_hys

            if act_pan is not None:
                count_act_pan[count_act_pan == 0] = 1
                act_pan = act_pan / count_act_pan

            if act_hys is not None:
                count_act_hys[count_act_hys == 0] = 1
                act_hys = act_hys / count_act_hys

            # Fallbacks so all four MATLAB files are always produced.
            if act_pan is None and act_hys is not None:
                act_pan = act_hys
            if act_hys is None and act_pan is not None:
                act_hys = act_pan[:, ::3, ::3]
            if act_pan is None and act_hys is None:
                raise ValueError('The network did not return activations, so act_8 files cannot be saved.')

            prefix = dataset if len(ids) == 1 else '{}_{}'.format(dataset, id_)

            _save_mat(
                os.path.join(save_folder, '{}_{}_post6_PAN.mat'.format(prefix, CNN_model)),
                'post_pan',
                _to_matlab_image_layout(post_pan)
            )
            _save_mat(
                os.path.join(save_folder, '{}_{}_post6_HYS.mat'.format(prefix, CNN_model)),
                'post_hys',
                _to_matlab_image_layout(post_hys)
            )
            _save_mat(
                os.path.join(save_folder, '{}_{}_act_8_PAN.mat'.format(prefix, CNN_model)),
                'act_pan',
                _to_matlab_image_layout(act_pan)
            )
            _save_mat(
                os.path.join(save_folder, '{}_{}_act_8_HYS.mat'.format(prefix, CNN_model)),
                'act_hys',
                _to_matlab_image_layout(act_hys)
            )
