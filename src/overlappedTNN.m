function [X] = overlappedTNN(X,omega,opts)

% overlappedTNN: Tensor completion using Overlapped Tensor Nuclear Norm (OTNN) method
%
% Input:
%   X     - Input tensor with known and unknown elements
%   omega - Logical indices indicating the positions of known elements
%   opts  - Optional structure for customizing algorithm behavior
%
% Output:
%   X     - Completed tensor
% 
% 
% By Yuning Qiu, 2023
% Email: yuning.qiu.gd@gmail.com


dim = size(X);
Nm = 1;
N = length(dim);

tol = 1e-20; 
max_iter = 300;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;
alp = 0.1*ones(1,N);
lam = 100;
l = floor((N-1)/2);
 
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end
if isfield(opts, 'alp');         alp = opts.alp;              end
if isfield(opts, 'lambda');      lam = opts.lambda;           end
if isfield(opts, 'l');           l = opts.l;                  end
 

% initialize the matrx
T = X;
To = zeros(size(T));
To(omega) = 1;
Tov = ~To;

Mmat = zeros(dim);
Mmat(omega) = X(omega);
M = cell(1,N);
for k=1:N
    M{k} = Mmat;
end

% initialize the matrix
Pmat = zeros(dim);
P = cell(1,N);
for k=1:N
    P{k} = Pmat;
end

 

errorList = zeros(max_iter, 1);
for iter = 1 : max_iter
    % update Mk. In our paper, this is Zk.
    for k = 1:N
        XPk = t3unfold(-X+P{k}/mu,k,l);
        [Mk,~] = prox_tnn(-XPk, lam*alp(k)/mu); 
        Mk = t3fold(Mk, dim, k);
        M{k}=Mk;
    end
    
    % update X 
    Xk = X;
    
    Xtemp = zeros(dim);
    Msum = Xtemp;
    for k=1:N
        Xtemp = P{k} + mu*M{k} + Xtemp;
        Msum = M{k} + Msum;
    end
    Xtemp = 1/Nm*(To.*T) + Xtemp;
    X = (1/(1/Nm+N*mu)*To+1/(N*mu)*Tov).*Xtemp;
    
    dY = 1/N * Msum - X;
    chgX = max(abs(Xk(:) - X(:)));
    chg = max([chgX, max(abs(dY(:)))]);
 
    errorList(iter) = norm(X(:) - Xk(:))  / norm(Xk(:));
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', chg=' num2str(chg) ', realErr=' num2str(err) ',rse=' num2str(errorList(iter))]); 
        end
    end
 
 
    if chg < tol
        break;
    end 
    
    % update dual variables Pk
    for k = 1:N
        P{k} = P{k} + mu*(M{k}-X);
    end
    if mu< max_mu
        mu = rho*mu;
    else
        mu =max_mu;
    end

end

