function working02_PseudoInverse()

clear all;
close all;
clc;

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

% Degrade the image using convolution of degradation function and image
gMediate = conv2(f, PSF, 'valid');
gMediate(imgWidth, imgHeight) = 0;
figure; imshow(gMediate,[]);

% Noise - Gaussian using randn
n = 2 * randn(imgWidth, imgHeight);

% Add noise to degraded image
g = double(gMediate) + n;
figure; imshow(g,[]);

G = fftshift(fft2(g));
N = fftshift(fft2(n));
F = fftshift(fft2(f));

% Obtain the degradation function transform
H = (G - N)./ F;

% Wiener filter for different SNR (k) values
for i = 0.0001 : 0.0002 : 0.001
    WienerRestoreDisplay(H, G, i);
end

% Apply pseudorandom filter
PseudoInverse_RestoreDisplay(H, G);

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

function PseudoInverse_RestoreDisplay(H, G)
    Ha = abs(H);
    Hb = 1./Ha;
    Hb(Hb > 1/0.0025) = 0; % Here k = 0.0025 - to remove very high values
    Fcap = G ./ Ha;
    RestoredFT = Fcap;
    RestoredImage = ifft2(ifftshift(RestoredFT));
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
