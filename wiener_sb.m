clear all;
close all;

% Main image
image = imread('cameraman.tif');
f = image;
imshow(image, []);

imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;
imgHeight = imgInfo.Height;

% Degradation function
hm = fspecial('motion', imgWidth, 0);
% center = hm(1, 5)

M = imgWidth;
N = imgHeight;
LEN = 9;
THETA = 0;
PSF = fspecial('motion', LEN, THETA); % Point Spread Function
%OTF  = psf2otf(PSF,[M N]); % PSF --> OTF
OTF = fft2(PSF, M, N); % Optical Transfer Function
figure, imshow(fftshift(abs(OTF)));
title('Optical Transfer Function')

h = imfilter(f, PSF);
figure; imshow(h, []);

% % Zero padding
% P = 2 * imgWidth;
% Q = 2 * imgHeight;
% padded = zeros(P);
% for i = 1:imgWidth
%     for j=1:imgHeight
%         padded(i,j) = image(i,j);
%     end
% end
% imshow(padded,[]);
% FT_Padded = fftshift(fft2(padded));


F = fftshift(fft2(f));
% Blur1 = FT_Padded * F;
% imshow(Blur1,[]);

% H = fftshift(fft2(h));
H = OTF; 

%n = imnoise(f, 'gaussian', 0, 0.01);
n = 0.5 * randn(imgWidth, imgHeight);
N = fftshift(fft2(n));

figure; imshow(n, []);
% Degraded function
%g = conv2(f, h, 'valid') + n;
%G = fftshift(fft2(g));
G = H .* F + N;
figure; imshow(ifft2(G), []);

% Wiener Filter as per Prof PDF
% x1 = 1./H;
% x2 = mean(mean(abs(N)^2));
% x3 = ((abs(H)^2) * (abs(F)^2));
% x4 = x2./x3;
% x5 = 1./(1 + x4);
% Fcap = (x1.* x5).*G;

x1 = 1./H;
x2 = abs(H).^2;
x3 = 0.00001; %abs(N).^2 / abs(F).^2;
% Weiner filter
Fcap = (x1.*(x2./(x2 + x3))).*G;

RestoredFT = Fcap;
RestoredImage = ifft2(ifftshift(RestoredFT));
figure; imshow(abs(RestoredImage), []); 

% Icap = ifft2(ifftshift(F));
% figure; imshow(abs(Icap), []); 


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
