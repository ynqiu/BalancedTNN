function T = latentTNN(Y,omegaIndex,opts)

% latentTNN: Tensor completion using Latent Tensor Nuclear Norm (LTNN) method
%
% Input:
%   Y           - Input tensor with known and unknown elements
%   omegaIndex  - Indices of known elements
%   opts        - Optional structure for customizing algorithm behavior
%
% Output:
%   T           - Completed tensor
% 
% 
% By Yuning Qiu, 2023
% Email: yuning.qiu.gd@gmail.com


dim       = size(Y);
N         = length(dim);

% Default parameters
tol       = 1e-10; 
maxIter   = 300; 
rho       = 1;
eta       = 1e-2;
maxMu     = 1e10;
DEBUG     = 0;
alp       = 0.1*ones(1,N);
lam       = 100;
tau       = ones(1,N);
gamma     = 1;
l         = floor((N-1)/2);
 
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol      = opts.tol;              end
if isfield(opts, 'max_iter');    maxIter  = opts.max_iter;         end
if isfield(opts, 'rho');         rho      = opts.rho;              end
if isfield(opts, 'mu');          eta      = opts.mu;               end
if isfield(opts, 'max_mu');      maxMu    = opts.max_mu;           end
if isfield(opts, 'DEBUG');       DEBUG    = opts.DEBUG;            end
if isfield(opts, 'alp');         alp      = opts.alp;              end
if isfield(opts, 'lambda');      lam      = opts.lambda;           end
if isfield(opts, 'tau');         tau      = opts.tau;              end
if isfield(opts, 'gamma');       gamma    = opts.gamma;            end
if isfield(opts, 'l');           l        = opts.l;                end
 
T                 = Y;
Omega             = zeros(dim);
Omega(omegaIndex) = 1;
onesTen           = ones(dim);

% latent components
Jmat        = zeros(dim);
Jmat(omegaIndex) = T(omegaIndex);
J           = cell(1,N);
for k=1:N
    J{k} = 1/N*Jmat;
end

% dual variable
P         = zeros(dim);

% error vector
errorList = zeros(maxIter, 1);

for iter   = 1:maxIter
    
    %% update Jk
    for k  = 1:N
        Jadj  = zeros(dim);
        for j = 1:N
            if j ~= k
                Jadj = Jadj + J{j};
            end
        end
        
        tau(k)     = eta*(N/(2-gamma) - 1) + 1e-4;
        TPJk       = t3unfold(T-P/eta-Jadj, k, l);
        Jkk        = t3unfold(J{k}, k, l);
        num        = eta*TPJk + tau(k)*Jkk;
        den        = eta + tau(k);
        % proximal operator
        Jkk        = prox_tnn(num/den, lam*alp(k)/(eta+tau(k))); 
        Jk         = t3fold(Jkk, dim, k);
        J{k}       = Jk;
    end
    
    %% update T
    Tpre    = T;
    Jsum    = Jadj + J{k};
    num     = P + eta*Jsum + Omega.*Y;
    den     = eta*onesTen+Omega;
    T       = num ./ den;
    
    %% convergence conditions
    errorList(iter) = norm(T(:) - Tpre(:))  / norm(Tpre(:));
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            err             = norm(T(:)-Y(:))/norm(Y(:));
            disp(['iter ' num2str(iter) ', eta=' num2str(eta) ...
                    ', ERR=' num2str(err) ',relCha=' num2str(errorList(iter))]); 
        end
    end
    
    % stop criterion
    if errorList(iter) < tol
        break;
    end 
    
    % update dual variables P
    P = P + gamma*eta*(Jsum-T);
    
    if eta< maxMu
        eta = rho*eta;
    else
        eta =maxMu;
    end

end
 