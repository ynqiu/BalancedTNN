function [Yk] = t3unfold(X, k, l)

%% old function
% unfold tensor X \in I_1*I_2*...*I_N to (I_1I_2...I_k) * (I_k+1 ... I_N) * I_t
%   Detailed explanation goes here

% compute the required parameters
% d = size(X);
% % d = [d, d(1)];
% N = numel(d);
% l = round((N-1)/2);
% 
% % compute t
%  if k> l
%      t = k-l;
%  else
%      t = k-l+N;
%  end
%  
%  % permute tensor
%  if k < l
%      if isempty(d(t+1:N))
%         Im = prod(d(1:k));
%      else   
%         Im = prod([d(t+1:N) d(1:k)]);
%      end
%      In = prod(d(k+1:t-1));
%      It = d(t);
%  else
%      if t==N
%         Im = prod(d(1:k));
%         In = prod(d(k+1:N-1));
%      else
%         Im = prod(d(t+1:k));
%         In = prod([d(k+1:N), d(1:t-1)]);
%      end
%      It = d(t);
%  end
%  
%  Y = permute(X, [t+1:N, 1:t]);
%  Yk = reshape(Y, Im, In, It);
 
 %% new function
% unfolding the high-order tensor to the third-order one
% by yuning.qiu
% e-mail: yuning.qiu.gd@gmail.com

xSize  = size(X);
N      = length(xSize);
% l      = floor((N-1)/2);
Xk     = shiftdim(X,k);
xSizek = size(Xk);
Yk     = reshape(Xk, [prod(xSizek(1:l)), prod(xSizek(l+1:N-1)), xSizek(N)]);

 
end