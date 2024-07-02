clear;
addpath(genpath('evaluation'));
addpath(genpath('lib'));
addpath(genpath('src'));

% List of available datasets
dataName = {'greek','boxes','dishes','kitchen','museum','tower','table','platonic'};
imNo = input('Select dataset (1-8): ');

% Set parameters
SR = input('Input sampling rate (e.g. 0.1): ');
c  = 0;
lambda = 250;



rng(2020, 'twister'); % Set random seed for reproducibility

% Load data
foldDir = 'data/lightfield/';
imDir = [foldDir, dataName{imNo}, '_128.mat'];
load(imDir);

X = T;
xSize = size(X);

% Generate random missing entries
Omega = zeros(xSize);
omegaIndex = randperm(prod(xSize), round(SR * prod(xSize)));
Omega(omegaIndex) = 1;
Xo = Omega .* X;
Ndim = size(Xo);
N = length(Ndim);

% Generate noise tensor
nObv = length(omegaIndex);
variance = c * norm(X(:), 'fro') / sqrt(prod(Ndim));
noiseVec = (variance) .* randn(nObv, 1);
noiseTen = zeros(Ndim);
noiseTen(omegaIndex) = noiseVec;
XoNoise = Xo + noiseTen;
XNoise = X + noiseTen;

% LTNN method

% Set options for LTNN
opts = [];
opts.DEBUG = 1;
opts.max_iter = 500;
opts.tol = 1e-8;
opts.max_mu = 1e10;
opts.mu = 1e-4;
opts.rho = 1.1;
opts.conv = true;

opts.alp = ones(1, N) / N;
opts.lambda = lambda;

% Run LTNN
tic;
[Xhat, conv_out] = latentTNN(XoNoise, omegaIndex, X, opts);
e_time = toc;

% Evaluate results
X4 = reshape(X, [128, 128, 3, 81]);
Xhat4 = reshape(Xhat, [128, 128, 3, 81]);

e_psnr_arr = videoPSNR(Xhat4, X4);
e_rmse_arr = videoPerfscore(Xhat4, X4);
e_ssim_arr = video_ssim_index(Xhat4, X4);
e_psnr = mean(e_psnr_arr);
e_rmse = mean(e_rmse_arr);
e_ssim = mean(e_ssim_arr);

% Print results
fprintf('LTNN Results:\n');
fprintf('PSNR: %.2f\n', e_psnr);
fprintf('RMSE: %.4f\n', e_rmse);
fprintf('SSIM: %.4f\n', e_ssim);
fprintf('Time: %.2f seconds\n', e_time);
