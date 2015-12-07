function trainFundusCNN(varargin)

%setup ;

% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('data/fundusdb3.mat') ;

% Visualize some of the data
figure(10) ; clf ; colormap gray ;
subplot(1,2,1) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==1)) ;
axis image off ;
title('training fundus for dr') ;

subplot(1,2,2) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==2)) ;
axis image off ;
title('validation fundus for dr') ;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = initFundusCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 3 ;
trainOpts.numEpochs = 40 ;
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.learningRate = 0.005 ;
%trainOpts.errorFunction = 'binary';
trainOpts.expDir = 'data/fundus-experiment' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imdb = load('data/fundusdb3.mat') ;
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if numel(trainOpts.gpus) > 0
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if numel(trainOpts.gpus) > 0
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/funduscnn.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
% net = load('data/funduscnn.mat') ;
% %net = load('data/chars-experiment/charscnn-jit.mat') ;
% 
% % Load the sentence
% im = im2single(imread('data/sentence-lato.png')) ;
% im = 256 * (im - net.imageMean) ;
% 
% % Apply the CNN to the larger image
% res = vl_simplenn(net, im) ;
% 
% % Visualize the results
% figure(3) ; clf ;
% decodeCharacters(net, imdb, im, res) ;

% -------------------------------------------------------------------------
% Part 4.6: train with jitter
% -------------------------------------------------------------------------

% trainOpts.batchSize = 100 ;
% trainOpts.numEpochs = 15 ;
% trainOpts.continue = true ;
% trainOpts.learningRate = 0.001 ;
% trainOpts.expDir = 'data/chars-jit-experiment' ;
% 
% % Initlialize a new network
% net = initializeCharacterCNN() ;
% 
% % Call training function in MatConvNet
% [net,info] = cnn_train(net, imdb, @getBatchWithJitter, trainOpts) ;
% 
% % Move the CNN back to CPU if it was trained on GPU
% if numel(trainOpts.gpus) > 0
%   net = vl_simplenn_move(net, 'cpu') ;
% end
% 
% % Save the result for later use
% net.layers(end) = [] ;
% net.imageMean = imageMean ;
% save('data/chars-experiment/charscnn-jit.mat', '-struct', 'net') ;
% 
% % Visualize the results on the sentence
% figure(4) ; clf ;
% decodeCharacters(net, imdb, im, vl_simplenn(net, im)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
im = 256 * reshape(im, 2048, 2048, 1, []);
labels = imdb.images.label(1,batch);

% --------------------------------------------------------------------
% function [im, labels] = getBatchWithJitter(imdb, batch)
% % --------------------------------------------------------------------
% im = imdb.images.data(:,:,batch) ;
% labels = imdb.images.label(1,batch) ;
% 
% n = numel(batch) ;
% train = find(imdb.images.set == 1) ;
% 
% sel = randperm(numel(train), n) ;
% im1 = imdb.images.data(:,:,sel) ;
% 
% sel = randperm(numel(train), n) ;
% im2 = imdb.images.data(:,:,sel) ;
% 
% ctx = [im1 im2] ;
% ctx(:,17:48,:) = min(ctx(:,17:48,:), im) ;
% 
% dx = randi(11) - 6 ;
% im = ctx(:,(17:48)+dx,:) ;
% sx = (17:48) + dx ;
% 
% dy = randi(5) - 2 ;
% sy = max(1, min(32, (1:32) + dy)) ;
% 
% im = ctx(sy,sx,:) ;
% 
% % Visualize the batch:
% % figure(100) ; clf ;
% % vl_imarraysc(im) ;
% 
% im = 256 * reshape(im, 32, 32, 1, []) ;



