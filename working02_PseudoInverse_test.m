function working02_PseudoInverse()

    clear;
    close all;
    clc;
    
    % constants
    sigma = 1;
    %weinerK = 0.0004;
    
    threshold = 0.025;

    % Main image
    image = imread('cameraman.tif');
    f = double(image);
    imgInfo = imfinfo('cameraman.tif');
    imgWidth = imgInfo.Width;
    imgHeight = imgInfo.Height;
    figure; imshow(image, []);
    
    % take another image
    f1 = double(imread('img0.tiff'));
    f2 = double(imread('img1.tiff'));
    f3 = double(imread('img2.tiff'));
    
    % Zero pad the image
    f = padding(f, imgWidth, imgHeight);
    imgWidth = imgWidth * 2;
    imgHeight = imgHeight * 2;
    displayTransformed(f);

    % Degradation function
    PSF = fspecial('motion', 15, 0);

    % Noise - Gaussian using randn
    n = sigma * randn(imgWidth, imgHeight);
    
    weinerK1 = sum(n(:).^2)/sum(f(:).^2);
    weinerK2 = sum(n(:).^2)/sum(f1(:).^2);
    weinerK3 = sum(n(:).^2)/(sum(f1(:).^2) + sum(f2(:).^2) + sum(f3(:).^2))/3;

    N = fftshift(fft2(n));
    F = fftshift(fft2(f));

    % Obtain the degradation function transform
    H = fftshift(fft2(PSF, imgWidth, imgHeight));
    G = H.*F + N;

    figure, imshow(abs(ifft2(ifftshift(G))),[]);

    % Wiener filter for different SNR (k) values
    % WienerRestoreDisplay(H, G, 0.00001);
    % WienerRestoreDisplay(H, G, 0.00003);
    % WienerRestoreDisplay(H, G, 0.00005);
     WienerRestoreDisplay(H, G, weinerK1);
     
     % Wiener restoration with different image
     WienerRestoreDisplay(H, G, weinerK2);
    
     % Wiener restoration with some different images
     WienerRestoreDisplay(H, G, weinerK3);
    

    % Apply pseudorandom filter
     PseudoInverse_RestoreDisplay(H, G, threshold);

end

function WienerRestoreDisplay(H, G, k)
    x1 = 1./H;
    x2 = abs(H).^2;
    x3 = k ;

    % Weiner filter
    Fcap = (x1.*(x2./(x2 + x3))).*G;
    RestoredFT = Fcap;
    RestoredImage = ifft2(ifftshift(RestoredFT));
    figure; imshow(abs(RestoredImage), []);
    %s = strcat('Wiener with NSR: ',
    title(strcat('Wiener with NSR: ' , sprintf('%0.5e',k)));
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
