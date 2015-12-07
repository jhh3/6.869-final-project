
clc;
clear all;
close all;

% Preprocessor settings
Scale = 1;
Filter = false;
Should_plot = false;
g = fspecial('gaussian', [25,25], 15);
Image_dir = 'fundus-all/images/';
Mask_dir = 'fundus-all/manual1/';
Processed_dir = 'fundus-all/processed2/';
h=2336
w=3504

left = (w - 2978) / 2;
top = (h - 1986) / 2;
neww = 2978;
newh = 1986;

% Iterate over images

image_files = dir(strcat(Image_dir,'*.jpg'));
mask_files = dir(strcat(Mask_dir,'*.tif'));

for i = 1:length(image_files')
    im_name = image_files(i).name
    mask_name = mask_files(i).name;
    
    fundus_im = imresize(imread(strcat(Image_dir, im_name)), Scale);
    fundus_mask = imresize(imread(strcat(Mask_dir, mask_name)), Scale);
    
    % Remove blood vessels and normalize intensity
    fundus_processed_im = preprocessFundus(fundus_im, Filter, fundus_mask, g);
    
    %crop and save jittered images
    [new_name, rem] = strtok(im_name, '.');
    extension = strtok(rem, '.');
    
    fundus_crop1 = imcrop(fundus_processed_im, [left top neww newh]);
    imwrite(fundus_crop1, ...
        strcat(Processed_dir, new_name, '_p1', rem), ...
        extension);
    
    fundus_crop2 = imcrop(fundus_processed_im, [(left-50) (top-25) neww newh]);
    imwrite(fundus_crop2, ...
        strcat(Processed_dir, new_name, '_p2', rem), ...
        extension);
    
    fundus_crop3 = imcrop(fundus_processed_im, [(left+35) (top-70) neww newh]);
    imwrite(fundus_crop3, ...
        strcat(Processed_dir, new_name, '_p3', rem), ...
        extension);
    
    fundus_crop4 = imcrop(fundus_processed_im, [(left+75) (top+50) neww newh]);
    imwrite(fundus_crop4, ...
        strcat(Processed_dir, new_name, '_p4', rem), ...
        extension);
    
    fundus_crop5 = imcrop(fundus_processed_im, [(left-60) (top+100) neww newh]);
    imwrite(fundus_crop5, ...
        strcat(Processed_dir, new_name, '_p5', rem), ...
        extension);
    
    fundus_crop6 = imcrop(fundus_processed_im, [(left-80) (top+40) neww newh]);
    imwrite(fundus_crop6, ...
        strcat(Processed_dir, new_name, '_p6', rem), ...
        extension);
    
    if Should_plot
        % Plot
        figure(1);imshow(fundus_im);title('Input Image');
        figure(2);imshow(fundus_processed_im);title('Image with Blood Vessels Extracted');
    end
end

