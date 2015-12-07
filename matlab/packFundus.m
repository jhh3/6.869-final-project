function packFundus()
% Put the fundus images into the proper format for MatConvNet

Processed_dir = 'fundus-all/processed/';
fundus = dir(strcat(Processed_dir,'*.jpg'));
image_types = 'dgh' ;

image_count = 15;
type_count = 3;

im = cell(image_count, type_count);
labels = cell(image_count, type_count);

file_count = 1;
for i = 1:image_count
  for j = 1:type_count
    im{i,j} = imread(fullfile('fundus-all', 'processed', fundus(file_count).name)) ;
    im{i,j} = rgb2gray((imresize(im{i,j}, [2048, 2048]))) ;
    im{i,j} = im2single(im{i,j}) ;    
    labels{i,j} = j ;
    file_count = file_count + 1;
  end
end

imdb.meta.classes = image_types ;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.fundus = 1:15 ;
imdb.images.id = 1:numel(im) ;
imdb.images.data = cat(3, im{:}) ;
imdb.images.label = cat(2, labels{:}) ;

% Create training and validation sets
sets = ones(image_count, type_count) ;
sets(end-4:end,:) = 2 ;
imdb.images.set = sets(:)' ;

save('data/fundusdb3.mat', '-v7.3', '-struct', 'imdb') ;
