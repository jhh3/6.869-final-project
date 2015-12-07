function im_p = preprocessFundus(im, do_filter, mask, g)

if nargin < 3
    %Create mask from extracted blood vessels if none passed
    im_gray = rgb2gray(im);
    Threshold = 7;
    mask = VesselExtract(im_gray, Threshold); 
end

if nargin < 4
    g = fspecial('gaussian', [5,5], 15);
end

im_mask = imfilter(mask, g);
im_mask(im_mask>0) = 255;
im_mask = im2bw(im_mask);

figure; imshow(

% Remove blood vessels and iterpolate based on mask
if ndims(im) == 3
    im_h2_r = regionfill(im(:,:,1), im_mask);
    im_h2_g = regionfill(im(:,:,2), im_mask);
    im_h2_b = regionfill(im(:,:,3), im_mask);

    im_p = cat(3, im_h2_r, im_h2_g, im_h2_b);    
else
    im_p = regionfill(im, im_mask);
end

if do_filter
    % Apply homomorphic filter for intensity normalization
    im_ph = homomorphic(im_p, 1.5, .25, 2, 0, 3);

    if ndims(im) == 3
        Yr=normalize8(im_ph(:,:,1));
        Yg=normalize8(im_ph(:,:,2));
        Yb=normalize8(im_ph(:,:,3));
        im_ph = cat(3, Yr, Yg, Yb);
    end

    im_p = uint8(im_ph);
end


