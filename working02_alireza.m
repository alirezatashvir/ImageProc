clear all;
close all;
clc;

% constants
sigma = 2;

% Main image
image = imread('cameraman.tif');
f = double(image);
imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;

displayTransformed(f);

% Degradation function
hm = fspecial('motion', 15, 0);

gMediate = imfilter(f,hm);
figure; imshow(gMediate,[]);

%n = wgn(imgWidth, imgWidth, 40);
n = sigma * randn(imgWidth, imgWidth);

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



% Define power spectrum of corrupted image

% first we should find the power spectrum of the crrupted image
% then we should extrapolate the power spectrum of the frequency and find
% the N

% first should sample the freq

power = log10(abs(fft2(g)).^2);
freq = log10(g);
figure; plot(freq,power(:,1));
