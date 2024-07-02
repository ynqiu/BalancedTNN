function [Y] = t3fold(X,d,k)
%%  old function
% % 
% % fold third-order tensor into high-order one
% % X: third-order tensor;
% % d: the shape of N order tensor
% % k: the kth mode unfold
% 
% % compute the required parameters
% N = numel(d);
% l = round((N-1)/2);
% 
% % compute t
%  if k>l
%      t = k-l;
%  else
%      t = k-l+N;
%  end
%  
%  % permute tensor
%  if k < l
%      if isempty(d(t+1:N))
%         Im = (d(1:k));
%      else   
%         Im = ([d(t+1:N) d(1:k)]);
%      end
%      In = (d(k+1:t-1));
%      It = d(t);
%  else
%      if t==N
%         Im = (d(1:k));
%         In = (d(k+1:N-1));
%      else
%         Im = (d(t+1:k));
%         In = ([d(k+1:N), d(1:t-1)]);
%      end
%      It = d(t);
%  end
%  
%  Yk = reshape(X, [Im, In, It]);
%  Y = permute(Yk, [N-t+1:N, 1:N-t]);
%% new function
xSize    = d;
N        = length(xSize);
xSizek   = circshift(xSize,-k);
Xk3      = X;
Xk       = reshape(Xk3, xSizek);
Xk       = shiftdim(Xk,N-k);
Y       = Xk;
end