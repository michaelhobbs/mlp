% ********* LOCAL COPY ***********

load('2class.mat');

% ***********  PREPROCESSING TRAINING AND VALIDATION DATA *****************
% 1) Use the normalization described in the handout page 5.
%    This shifts and rescales data to mean=0 and variance=1.
%    NOTE: If we finish everything else early, we could look into better
%    normalization methods.
% 2) (Not really a second step) Perform normalization on training dataset,
%    use training dataset mean and variance to normalize testing dataset.

[dlt1,dlv1,tdl1] = preprocess_data(dlt1,dlv1,tdl1);
[drt1,drv1,tdr1] = preprocess_data(drt1,drv1,tdr1);


% ********************* MLP FOR BINARY CLASSIFICATION *********************

%% CONSTANTS:
H1 = 2;
H2 = 4;
size_left = size(dlt1,2);
size_right = size(drt1,2);
nu_start = 0.5;
pi = 0.5;
num_batches=10;
% num_epochs = 15; % when this is too high, we overfit the network (ex:error(25) > error(5))?
error = [];
incorrect_rate = [] ;

batch_size = size(lv1,1)/num_batches; % take care to make num_batches such that number of samples is a multiple of it (we could make this better by treating the last batch on its own, it would have fewer samples in it than the other batches)

% cmap = distinguishable_colors(num_epochs);
%error_epoch = zeros(num_epochs,1);
%incorrect_epoch = zeros(num_epochs,1);
%incorrect_rate = zeros(num_batches,1);

layers = [(size_left + size_right)*H1, 2*H1*2*H2, H2];
disp(['Network size: ' num2str(layers)]);

%initialize network:
 network = init_network(size_left,size_right,H1,H2); %will hold W and Bias vectors.
 
 %inital error:
 [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
error(1) = mean(log(ones(size(lv1,1),1)+exp(-lv1.*a3')));
 disp(['Error before training: ' num2str(error(1))]);
 incorrect_rate(1) = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
    disp(['Mispredicted percentage at start : ' num2str(incorrect_rate(1))]);

%% Backpropagation:
network_delta = init_network_delta(H1,H2,size_left,size_right);

k=1;
% for k=1:num_epochs % at each epoch make a random permutation of entire training set
    
while ( k==1 || error(k)<error(k-1) )
    %nu = nu_start*exp(-k); % exponential decay is bad
    nu=nu_start/k;
    rand_epoch = randperm(size(dlt1,1));
     for i=1:num_batches % in each epoch send random permuted data through in batches
             rand_batch = rand_epoch((i-1)*batch_size+1:i*batch_size);
             data_left = dlt1(rand_batch,:);
             data_right = drt1(rand_batch,:);
             labels_left = lt1(rand_batch,:);
             [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(data_left, data_right, network);
             [network, network_delta] = backprop(network, labels_left, data_left, data_right, nu, pi, network_delta, a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2);
     end
     [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);

     error(k+1)= mean(-lv1.*a3'+log(ones(size(lv1,1),1)+exp(lv1.*a3'))); 

     incorrect_rate(k+1) = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
     if mod(k,100)==0
        disp(['Validation error at epoch ' num2str(k) ' : ' num2str(error(k+1))]);
     end
     %disp(['Validation miss rate at epoch ' num2str(k) ' : ' num2str(incorrect_rate(k+1))]);
k=k+1;
end
% hold off;
plot([0:k-1],error);
% figure(2);
% plot([0:num_epochs],incorrect_rate); xlabel('epoch');ylabel('miss rate');title('miss rate on validation set by epoch');
% figure(2);
% plot([1:num_epochs],error_epoch); xlabel('epoch'); ylabel('error'); title('Error at each epoch');
% 
% figure(3);
% plot([1:num_epochs],incorrect_epoch); xlabel('epoch'); ylabel('mispredicted %'); title('Percentage of mispredicted images et each epoch');

% final test of network quality on test data:
% [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(tdl1,tdr1,network);
% error_final = mean(log(ones(size(tl1,1),1)+exp(-tl1.*a3'))); 
% disp(['Error at end on test data : ' num2str(error_final)]);
% incorrect_rate_final = sum(sign(tl1)~=sign(a3'))/size(tl1,1)*100;
% disp(['Mispredicted percentage at end on test data : ' num2str(incorrect_rate_final)]);