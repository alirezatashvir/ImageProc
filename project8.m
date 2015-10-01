
image = imread('cameraman.tif');
f = image;
h = fspecial('motion');
figure; imshow(image,[]);
imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;
imgHeight = imgInfo.Height;
n = wgn(imgHeight, imgWidth,0)
g = conv2(double(h),double(f))

figure; imshow(g,[]);
% noisedImage = imgaussfilt(blurred,1);

%noised = fspecial('gaussian', [3 3], 1);
%noisedImage = imfilter(blurred,noised,'replicate');
% figure; imshow(noisedImage,[]);
% noise2 = imnoise(blurred,'gaussian', 0 , 0.1);
% figure; imshow(noise2,[]);
% noise3 = imnoise(blurred,'gaussian',0,0.01);
% figure; imshow(noise3,[]);
%H = fft2(image);

%HConj = conj(H);

%% 

imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width
imgHeight = imgInfo.Height
%% 
