function working02_PseudoInverse()

clear all;
close all;
clc;

NSR = 0.0008;
SIGMA = 5;

% Main image
image = imread('cameraman.tif');
f = double(image);
imgInfo = imfinfo('cameraman.tif');
imgWidth = imgInfo.Width;
imgHeight = imgInfo.Height;
figure; imshow(image, []);

% Zero pad the image
f = padding(f, imgWidth, imgHeight);
imgWidth = imgWidth * 2;
imgHeight = imgHeight * 2;
displayTransformed(f);

% Degradation function
PSF = fspecial('motion', 15, 0);

% Noise - Gaussian using randn
n = SIGMA * randn(imgWidth, imgHeight);

N = fftshift(fft2(n));
F = fftshift(fft2(f));

% Obtain the degradation function transform
H = fftshift(fft2(PSF, imgWidth, imgHeight));
G = H.*F + N;

figure; imshow(ifft2(ifftshift(G)), []);
title('In Fourier domain');

% Use convolution of PSF and original image
Hf = conv2(f, PSF, 'valid');
Hf(imgWidth, imgHeight) = 0;
FFT_Hf = fftshift(fft2(Hf));
G1 = FFT_Hf + N;

figure; imshow(ifft2(ifftshift(G1)), []);
title('In Image Domain - Using conv2');

% Wiener filter for different NSR (k) values
WienerRestoreDisplay(H, G, NSR);
title('Wiener Filter - NSR constant');

estimated_nsr = sum(n(:).^2)/sum(f(:).^2); % Perseval theorem
WienerRestoreDisplay(H, G, estimated_nsr);
title('Wiener Filter - NSR Perseval theorem');

% Apply pseudorandom filter
PseudoInverse_RestoreDisplay(H, G, 0.0143);
title('Pseudo Inverse Filter');

end

function WienerRestoreDisplay(H, G, k)
x1 = 1./H;
x2 = abs(H).^2;
x3 = k ;

% Weiner filter
Fcap = (x1.*(x2./(x2 + x3))).*G;
RestoredFT = Fcap;
RestoredImage = ifft2(RestoredFT);
figure; imshow(abs(RestoredImage), []);
end

function PseudoInverse_RestoreDisplay(H, G, threshold)
    Ha = abs(H);
    Hb = 1./Ha;
    Hb(Hb > 1/threshold) = 0; % Here k = 0.0025 - to remove very high values %%% or 0.0143
    Fcap = G .* Hb;
    RestoredFT = Fcap;
    RestoredImage = real(ifft2(ifftshift(RestoredFT)));
    figure; imshow(abs(RestoredImage), []);
end

function FT_Padded = padding(f, imgWidth, imgHeight)

% Zero padding
P = 2 * imgWidth;
Q = 2 * imgHeight;
padded = zeros(P);
for i = 1:imgWidth
    for j=1:imgHeight
        padded(i,j) = f(i,j);
    end
end
imshow(padded,[]);
FT_Padded = padded; %fftshift(fft2(padded));
end
