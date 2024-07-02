clear;
addpath(genpath('evaluation'));
addpath(genpath('lib'));
addpath(genpath('src'));

% Available datasets
dataName = {'hall','salesman','highway','silent'};
imNo = input('Select dataset (1-4): ');

% Set parameters
SR = 0.2;  
c = 0.05;
lambda = 2e+3;

rng(2020, 'twister');
frameNo = 100;

% Load data
foldDir = 'data/YUV/';
imDir = [foldDir, dataName{imNo}, 'RGB.mat'];
load(imDir);

X = T(:,:,:,1:100);
xSize = size(X);

% Generate random missing entries
Omega = zeros(xSize);
omegaIndex = randperm(prod(xSize), round(SR * prod(xSize)));
Omega(omegaIndex) = 1;
Xo = Omega .* X;
Ndim = size(Xo);

% Generate noise tensor
nObv = length(omegaIndex);
variance = c * norm(X(:), 'fro') / sqrt(prod(Ndim));
noiseVec = (variance) .* randn(nObv, 1);
noiseTen = zeros(Ndim);
noiseTen(omegaIndex) = noiseVec;
XoNoise = Xo + noiseTen;

% Reshape for high-order tensor completion
XoNoiseh = reshape(XoNoise, [12, 12, 11, 16, 3, frameNo]);
Omegah = reshape(Omega, [12, 12, 11, 16, 3, frameNo]);
omegaIndexh = find(Omegah == 1);
Ndimh = size(XoNoiseh);
Nh = length(Ndimh);

% LTNN method
opts = [];
opts.DEBUG = 1;
opts.max_iter = 500;
opts.tol = 1e-6;
opts.max_mu = 1e10;
opts.mu = 1e-4;
opts.rho = 1.1;
opts.alp = [1, 1, 1, 10, 1e-2, 5e+2];
opts.lambda = lambda;
 
tic;
Xhath = latentTNN(XoNoiseh, omegaIndexh, opts);
e_time = toc;
Xhat = reshape(Xhath, xSize);

% Evaluate results
e_psnr_arr = videoPSNR(Xhat, X);
e_rmse_arr = videoPerfscore(Xhat, X);
e_ssim_arr = video_ssim_index(Xhat, X);

e_psnr = mean(e_psnr_arr);
e_rmse = mean(e_rmse_arr);
e_ssim = mean(e_ssim_arr);

% Print results
fprintf('LTNN Results:\n');
fprintf('PSNR: %.2f\n', e_psnr);
fprintf('RMSE: %.4f\n', e_rmse);
fprintf('SSIM: %.4f\n', e_ssim);
fprintf('Time: %.2f seconds\n', e_time);

