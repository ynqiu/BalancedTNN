function psnr_array=videoPSNR(Xhat, Xim)
% test video psnr frame by frame

if ndims(Xhat)==4
    numFrame=size(Xhat,4);
    psnr_array=zeros(numFrame,1);
    for i=1:numFrame
        psnr_array(i)=PSNR_RGB(Xhat(:,:,:,i),Xim(:,:,:,i));
        %ssim_index(rgb2gray(uint8(Xhat(:,:,:,i))),rgb2gray(uint8(Xim(:,:,:,i))));
    end
elseif ndims(Xhat)==3
    numFrame=size(Xhat,3);
    psnr_array=zeros(numFrame,1);
    for i=1:numFrame
        psnr_array(i)=PSNR_RGB(Xhat(:,:,i),Xim(:,:,i));
        %ssim_index(rgb2gray(uint8(Xhat(:,:,i))),rgb2gray(uint8(Xim(:,:,i))));
    end   
end
end
