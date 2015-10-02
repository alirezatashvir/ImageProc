function tfImage = displayTransformed(image)

    absF = fftshift(log(1+abs(fft2(image))));
    figure; imshow(image, []);
    figure; imshow(absF, []);
    tfImage = absF;
end

