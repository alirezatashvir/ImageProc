function working03()

    clear all;
    close all;
    clc;

    idealNSR = 0.01;
    SIGMA = 0.01;
    PsudoThreshold = 0.025;
    Theta = 0;
    Len = 15;

    % Main image
    image = imread('cameraman.tif');
    f = mat2gray(image);
    imgInfo = imfinfo('cameraman.tif');
    imgWidth = imgInfo.Width;
    imgHeight = imgInfo.Height;
    figure; imshow(image, []);
    
    % take other images
    f1 = mat2gray(imread('img0.tiff'));
    f2 = mat2gray(imread('img1.tiff'));
    f3 = mat2gray(imread('img2.tiff'));

    % Zero pad the image
    f = padding(f, imgWidth, imgHeight);
    imgWidth = imgWidth * 2;
    imgHeight = imgHeight * 2;
    displayTransformed(f);

    % Degradation function
    PSF = fspecial('motion', Len, Theta);

    % Noise - Gaussian using randn
    n = SIGMA * randn(imgWidth, imgHeight);

    N = fftshift(fft2(n));
    F = fftshift(fft2(f));

    % Obtain the degradation function transform
    H = fftshift(fft2(PSF, imgWidth, imgHeight));
    G = H.*F + N;

    figure; imshow(real(ifft2(ifftshift(G))), []);
    title('In Fourier domain');

    % Use convolution of PSF and original image
    Hf = conv2(f, PSF, 'valid');
    Hf(imgWidth, imgHeight) = 0;
    FFT_Hf = fftshift(fft2(Hf));
    G1 = FFT_Hf + N;

    figure; imshow(real(ifft2(ifftshift(G1))), []);
    title('In Image Domain - Using conv2');

    % Wiener filter for different NSR (k) values
    WienerRestoreDisplay(H, G, idealNSR);
    title('Wiener Filter - Ideal NSR constant');

    estimated_nsr = sum(n(:).^2)/sum(f(:).^2); % Parseval theorem with the perfect image
    estimated_nsr2 = sum(n(:).^2)/sum(f2(:).^2); % Parseval theorem with one different image
    estimated_nsr3 = sum(n(:).^2)/((sum(f1(:).^2) + sum(f2(:).^2) + sum(f3(:).^2))/3); % Parseval theorem with some different images
    disp(['Power spectrum of perfect image ',num2str(sum(f(:).^2))]);
    disp(['Power spectrum of image one ',num2str(sum(f1(:).^2))]);
    disp(['Power spectrum of image two ',num2str(sum(f2(:).^2))]);
    disp(['Power spectrum of image three ',num2str(sum(f3(:).^2))]);
    disp(['Mean power spectrum of images ' , num2str((sum(f1(:).^2) + sum(f2(:).^2) + sum(f3(:).^2))/3)]);
    disp(['Estimated NSR of image two ', num2str(estimated_nsr2)]);
    disp(['Estimated NSR of all images ', num2str(estimated_nsr3)]);
    
    WienerRestoreDisplay(H, G, estimated_nsr);
    title('Wiener Filter - PS perfect image');
    
    WienerRestoreDisplay(H, G, estimated_nsr2);
    title('Wiener Filter - PS different image');
    
    WienerRestoreDisplay(H, G, estimated_nsr3);
    title('Wiener Filter - PS Multiple images');
    
    

    % Apply pseudorandom filter
    PseudoInverse_RestoreDisplay(H, G, PsudoThreshold);
    title('Pseudo Inverse Filter');

    end

    function WienerRestoreDisplay(H, G, k)
        x1 = 1./H;
        x2 = abs(H).^2;
        x3 = k ;

        % Weiner filter
        Fcap = (x1.*(x2./(x2 + x3))).*G;
        RestoredImage = ifft2(ifftshift(Fcap));
        % Unpad
        RestoredImage = Unpad(RestoredImage);
        figure; imshow(real(RestoredImage), []);
       
    end

    function PseudoInverse_RestoreDisplay(H, G, threshold)
        Ha = abs(H);
        Hb = 1./Ha;
        Hb(Hb > 1/threshold) = 0; % Here k = 0.0025 - to remove very high values %%% or 0.0143
        Fcap = G .* Hb;
        RestoredFT = Fcap;
        RestoredImage = real(ifft2(ifftshift(RestoredFT)));
        % Unpad
        RestoredImage = Unpad(RestoredImage);
        figure; imshow(RestoredImage, []);
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
        FT_Padded = padded;
end


function unpadded = Unpad(paddedImage)

    unpadded = paddedImage;
    [l, b] = size(paddedImage);
    unpadded(l/2:l, :) = [];
    unpadded(:, b/2:b) = [];

end

