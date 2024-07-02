function rse_array=videoPerfscore(Xhat, Xim)
    % test video re frame by frame

if ndims(Xhat)==4
    numFrame=size(Xhat,4);
    rse_array=zeros(numFrame,1);
    for i=1:numFrame
        rse_array(i)=perfscore(Xhat(:,:,:,i),Xim(:,:,:,i));
        %ssim_index(rgb2gray(uint8(Xhat(:,:,:,i))),rgb2gray(uint8(Xim(:,:,:,i))));
    end
elseif ndims(Xhat)==3
    numFrame=size(Xhat,3);
    rse_array=zeros(numFrame,1);
    for i=1:numFrame
        rse_array(i)=perfscore(Xhat(:,:,i),Xim(:,:,i));
        %ssim_index(rgb2gray(uint8(Xhat(:,:,i))),rgb2gray(uint8(Xim(:,:,i))));
    end   
end
end
