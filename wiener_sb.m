clear all;
close all;

% Main image
image = imread('cameraman.tif');
f = image;

% Degradation function
h = fspecial('motion', 9, 0);
h = imfilter(f, h,'replicate');
figure; imshow(h, []);

imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;
imgHeight = imgInfo.Height;

F = fftshift(fft2(f));
H = fftshift(fft2(h));
%n = imnoise(f, 'gaussian', 0, 0.01);
n = randn(imgWidth, imgHeight);
N = fftshift(fft2(n));

figure; imshow(n, []);
% Degraded function
g = conv2(f, h, 'valid') + n;
%G = H .* F + N;
G = fftshift(fft2(g));
figure; imshow(g, []);

% Wiener Filter as per Prof PDF
% x1 = 1./H;
% x2 = mean(mean(abs(N)^2));
% x3 = ((abs(H)^2) * (abs(F)^2));
% x4 = x2./x3;
% x5 = 1./(1 + x4);
% Fcap = x1.* x5.*G;

x1 = 1./H;
x2 = abs(H).^2;
x3 = 0.00001; %abs(N).^2 / abs(F).^2;
% Weiner filter
Fcap = (x1.*(x2./(x2 + x3))).*G;
Icap = ifft2(ifftshift(Fcap));
figure; imshow(abs(Icap), []); 

Icap = ifft2(ifftshift(F));
figure; imshow(abs(Icap), []); 


% % Inverse Filter
% Fcap = F + N/H;
% figure; imshow(Fcap, []); 
% Icap = ifft2(Fcap);
% figure; imshow(abs(Icap), []); 

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
