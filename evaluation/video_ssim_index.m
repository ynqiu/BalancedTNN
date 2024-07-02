function ssim_index_array = video_ssim_index(Xhat, Xim)
% test ssim frame by frame
if ndims(Xhat)==4
    numFrame=size(Xhat,4);
    ssim_index_array=zeros(numFrame,1);
    for i=1:numFrame
        ssim_index_array(i)=ssim_index(rgb2gray(uint8(Xhat(:,:,:,i))),rgb2gray(uint8(Xim(:,:,:,i))));
    end
elseif ndims(Xhat)==3
    numFrame=size(Xhat,3);
    ssim_index_array=zeros(numFrame,1);
    for i=1:numFrame
        ssim_index_array(i)=ssim_index((uint8(Xhat(:,:,i))),(uint8(Xim(:,:,i))));
    end
end

end