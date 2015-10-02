clear all;
close all;
clc;

% Main image
image = imread('cameraman.tif');
f = double(image);
imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;

displayTransformed(f);

% Degradation function
hm = fspecial('motion', 9, 0);

gMediate = imfilter(f,hm,'replicate');
figure; imshow(gMediate,[]);

n = wgn(imgWidth, imgWidth, 40);

g = double(gMediate) + n;
figure; imshow(g,[]);

G = fftshift(fft2(g));
displayTransformed(g);

N = fftshift(fft2(n));
F = fftshift(fft2(f));

H = (G-N)./ F;

% hf = conv2(f,hm, 'valid');

x1 = 1./H;
x2 = abs(H).^2;
x3 = 0.5;%abs(N).^2 ./ abs(F).^2;
% Weiner filter
Fcap = (x1.*(x2./(x2 + x3))).*G;

RestoredFT = Fcap;
RestoredImage = ifft2(ifftshift(RestoredFT));
figure; imshow(abs(RestoredImage), []); 

