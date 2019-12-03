from __future__ import division, absolute_import, print_function
import sys
import os
import argparse
import os.path
import random
import numpy as np
from PIL import Image
import scipy.misc
import numpy as np
from numpy.lib.arraypad import _validate_lengths
from scipy.ndimage import uniform_filter, gaussian_filter
from decimal import Decimal
from skimage import io
parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', default='', help="path to GT images")
parser.add_argument('--result_dir', default='', help="path to dehazed images")
opt = parser.parse_args()

dtype_range = {np.bool_: (False, True),
			   np.bool8: (False, True),
			   np.uint8: (0, 255),
			   np.uint16: (0, 65535),
			   np.uint32: (0, 2 ** 32 - 1),
			   np.uint64: (0, 2 ** 64 - 1),
			   np.int8: (-128, 127),
			   np.int16: (-32768, 32767),
			   np.int32: (-2 ** 31, 2 ** 31 - 1),
			   np.int64: (-2 ** 63, 2 ** 63 - 1),
			   np.float16: (-1, 1),
			   np.float32: (-1, 1),
			   np.float64: (-1, 1)}


def crop(ar, crop_width, copy=False, order='K'):
	ar = np.array(ar, copy=False)
	crops = _validate_lengths(ar, crop_width)
	slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
	if copy:
		cropped = np.array(ar[slices], order=order, copy=True)
	else:
		cropped = ar[slices]
	return cropped


def compare_ssim(X, Y, win_size=None, gradient=False,
				 data_range=None, multichannel=False, gaussian_weights=False,
				 full=False, dynamic_range=None, **kwargs):
	if not X.dtype == Y.dtype:
		raise ValueError('Input images must have the same dtype.')

	if not X.shape == Y.shape:
		raise ValueError('Input images must have the same dimensions.')

	if dynamic_range is not None:
		# warn('`dynamic_range` has been deprecated in favor of '
		#     '`data_range`. The `dynamic_range` keyword argument '
		#     'will be removed in v0.14', skimage_deprecation)
		data_range = dynamic_range

	if multichannel:
		# loop over channels
		args = dict(win_size=win_size,
					gradient=gradient,
					data_range=data_range,
					multichannel=False,
					gaussian_weights=gaussian_weights,
					full=full)
		args.update(kwargs)
		nch = X.shape[-1]
		mssim = np.empty(nch)
		if gradient:
			G = np.empty(X.shape)
		if full:
			S = np.empty(X.shape)
		for ch in range(nch):
			ch_result = compare_ssim(X[..., ch], Y[..., ch], **args)
			if gradient and full:
				mssim[..., ch], G[..., ch], S[..., ch] = ch_result
			elif gradient:
				mssim[..., ch], G[..., ch] = ch_result
			elif full:
				mssim[..., ch], S[..., ch] = ch_result
			else:
				mssim[..., ch] = ch_result
		mssim = mssim.mean()
		if gradient and full:
			return mssim, G, S
		elif gradient:
			return mssim, G
		elif full:
			return mssim, S
		else:
			return mssim

	K1 = kwargs.pop('K1', 0.01)
	K2 = kwargs.pop('K2', 0.03)
	sigma = kwargs.pop('sigma', 1.5)
	if K1 < 0:
		raise ValueError("K1 must be positive")
	if K2 < 0:
		raise ValueError("K2 must be positive")
	if sigma < 0:
		raise ValueError("sigma must be positive")
	use_sample_covariance = kwargs.pop('use_sample_covariance', True)

	if win_size is None:
		if gaussian_weights:
			win_size = 11  # 11 to match Wang et. al. 2004
		else:
			win_size = 7  # backwards compatibility

	if np.any((np.asarray(X.shape) - win_size) < 0):
		raise ValueError(
			"win_size exceeds image extent.  If the input is a multichannel "
			"(color) image, set multichannel=True.")

	if not (win_size % 2 == 1):
		raise ValueError('Window size must be odd.')

	if data_range is None:
		dmin, dmax = dtype_range[X.dtype.type]
		data_range = dmax - dmin

	ndim = X.ndim

	if gaussian_weights:
		# sigma = 1.5 to approximately match filter in Wang et. al. 2004
		# this ends up giving a 13-tap rather than 11-tap Gaussian
		filter_func = gaussian_filter
		filter_args = {'sigma': sigma}

	else:
		filter_func = uniform_filter
		filter_args = {'size': win_size}

	# ndimage filters need floating point data
	X = X.astype(np.float64)
	Y = Y.astype(np.float64)

	NP = win_size ** ndim

	# filter has already normalized by NP
	if use_sample_covariance:
		cov_norm = NP / (NP - 1)  # sample covariance
	else:
		cov_norm = 1.0  # population covariance to match Wang et. al. 2004

	# compute (weighted) means
	ux = filter_func(X, **filter_args)
	uy = filter_func(Y, **filter_args)

	# compute (weighted) variances and covariances
	uxx = filter_func(X * X, **filter_args)
	uyy = filter_func(Y * Y, **filter_args)
	uxy = filter_func(X * Y, **filter_args)
	vx = cov_norm * (uxx - ux * ux)
	vy = cov_norm * (uyy - uy * uy)
	vxy = cov_norm * (uxy - ux * uy)

	R = data_range
	C1 = (K1 * R) ** 2
	C2 = (K2 * R) ** 2

	A1, A2, B1, B2 = ((2 * ux * uy + C1,
					   2 * vxy + C2,
					   ux ** 2 + uy ** 2 + C1,
					   vx + vy + C2))
	D = B1 * B2
	S = (A1 * A2) / D

	# to avoid edge effects will ignore filter radius strip around edges
	pad = (win_size - 1) // 2

	# compute (weighted) mean of ssim
	mssim = crop(S, pad).mean()

	if gradient:
		# The following is Eqs. 7-8 of Avanaki 2009.
		grad = filter_func(A1 / D, **filter_args) * X
		grad += filter_func(-S / B2, **filter_args) * Y
		grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
							**filter_args)
		grad *= (2 / X.size)

		if full:
			return mssim, grad, S
		else:
			return mssim, grad
	else:
		if full:
			return mssim, S
		else:
			return mssim


# SCALE = 8
SCALE = 1


def output_psnr_mse(img_orig, img_out):
	squared_error = np.square(img_orig - img_out)
	mse = np.mean(squared_error)
	psnr = 10 * np.log10(1.0 / mse)
	return psnr


def _open_img(img_p):
	F = io.imread(img_p).astype(float) / 255.0
	h, w, c = F.shape
	F = F[:h - h % SCALE, :w - w % SCALE, :]
	boundarypixels = SCALE
	F = F[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
	return F


def _open_img_ssim(img_p):
	F = io.imread(img_p) # .astype(float)
	h, w, c = F.shape
	F = F[:h - h % SCALE, :w - w % SCALE, :]
	boundarypixels = SCALE
	F = F[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
	return F


def compute_psnr(ref_im, res_im):
	return output_psnr_mse(
		_open_img(os.path.join(ref_dir, ref_im)),
		_open_img(os.path.join(res_dir, res_im))
	)


def compute_mssim(ref_im, res_im):
	ref_img = _open_img_ssim(os.path.join(ref_dir, ref_im))
	res_img = _open_img_ssim(os.path.join(res_dir , res_im))
	channels = []
	for i in range(3):
		channels.append(compare_ssim(ref_img[:, :, i], res_img[:, :, i],
									 gaussian_weights=True, use_sample_covariance=False))
	return np.mean(channels)
 



res_dir = opt.gt_dir
ref_dir = opt.result_dir

runtime = -1
cpu = -1
data = -1
other = ""

ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
# if not (len(ref_pngs)==5 and len(res_pngs)==5):
# raise Exception('Expected 5 .png images, got %d'%len(res_pngs))

scores = []
scores_ssim = []
data = zip(ref_pngs, res_pngs)
for (ref_im, res_im) in np.array(list(data)):
    print(ref_im, res_im,'psnr:',compute_psnr(ref_im, res_im),'ssim:',compute_mssim(ref_im, res_im))
    scores.append(compute_psnr(ref_im, res_im))
    scores_ssim.append(compute_mssim(ref_im, res_im))
# print(ref_im, res_im)


# print(scores[-1])
psnr = np.mean(scores)
psnr = Decimal(psnr).quantize(Decimal('0.0000'))
mssim = np.mean(scores_ssim)
mssim = Decimal(mssim).quantize(Decimal('0.0000'))
print("\n psnr:\n", psnr,'\n compute ssim:\n',mssim)



