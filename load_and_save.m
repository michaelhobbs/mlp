%------------- Pattern Recognition and Machine Learning -------------------
%--------------------------------------------------------------------------
%------------------------- Miniproject ------------------------------------

% help : http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
%       
%
%% TODO:
% **************  LOADING AND SETTING UP THE DATASETS *********************
% 1) Load the data downloaded from moodle.
% 2) Plot histogram of labels and reshape/imshow a few samples.
% 3) Use randperm function on training dataset in order to randomly split 
%    it into training and validation sets. (2/3 samples in training, 1/3 in
%    validation) (NOTE: we do not need to touch the test dataset)
% 4) save these new datasets so that when we are trying our implementation
%    out, we will know any change in performance is due to something we did
%    rather than a new random permutation of the data samples.

% 1)
load('norb_binary.mat'); % cat=1||3; 900 samples of each
%load('norb_5class.mat');
% Data is of the form:
% test_cat_s : 1x5400 vector of labels (CATegories?)
% test_left_s : 576x5400 matrix where each row is a different data point in
%   the input space (a pixel for us) and each column is an instance of the 
%   data (an image) taken by left camera
%   i.e. we have 5400 samples of 24x24 images vectorized into 1x576
% test_right_s : 576x5400 matrix from right camera
% same again for test data (same number of samples: 5400)%
% NOTE: both datasets have variables of the same name, load one at a time
% and save them under different names.

% 2)
% figure(1);
% hist(test_cat_s,[0:4]);
% title('initial permutation of data');
% figure;
% imshow(reshape(test_left_s(:,1),24,24));

% 3)
class_training_l = [train_left_s'];
class_training_r = [train_right_s'];
size_data = size(class_training_l,1);
shuffle = randperm(size_data);
rand_training_l = class_training_l(shuffle,:);
rand_training_r = class_training_r(shuffle,:);
rand_cat = train_cat_s(shuffle)';



dl1 = rand_training_l;
dr1 = rand_training_r;
l1 = rand_cat;


% split so that 1/3 is validation, 2/3 is pure training
dlt1 = dl1(1:round(2*size_data/3),:);
dlv1 = dl1(round(2*size_data/3)+1:end,:);

drt1 = dr1(1:round(2*size_data/3),:);
drv1 = dr1(round(2*size_data/3)+1:end,:);

lt1 = l1(1:round(2*size_data/3),:);
lv1 = l1(round(2*size_data/3)+1:end,:);

%make test datasets:
tdl1 = test_left_s';
tdr1 = test_right_s';
tl1 = test_cat_s';

%we should probably use short names for our variables like:
%d1l: dataset 1 (aka binary classifier), left camera
%d1r, l1: labels of dataset 1
%d2l,d2r,l2: dataset 2 (5 classes)
%td1l:test dataset 1
%tl1: labels of test dataset 1
%tl2,td2l,td2r,td1r

% set labels to -1 and +1
lt1(lt1==3) = -1;
lv1(lv1==3) = -1;
tl1(tl1==3) = -1;

% 4)
save('2class.mat','dlt1','dlv1','drt1','drv1','lt1','lv1','tdl1','tdr1','tl1');