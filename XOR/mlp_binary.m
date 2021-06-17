load('2class.mat');

% ***********  PREPROCESSING TRAINING AND VALIDATION DATA *****************
% 1) Use the normalization described in the handout page 5.
%    This shifts and rescales data to mean=0 and variance=1.
%    i.e. Perform normalization on training dataset,
%    use training dataset mean and variance to normalize testing dataset.

[dlt1,dlv1,tdl1] = preprocess_data(dlt1,dlv1,tdl1);
[drt1,drv1,tdr1] = preprocess_data(drt1,drv1,tdr1);


% ********************* MLP FOR BINARY CLASSIFICATION *********************

%% CONSTANTS:
H1 = 1;
H2 = 20;
size_left = size(dlt1,2);
size_right = size(drt1,2);
nu = 0.001;
pi = 0;

layers = [(size_left + size_right)*H1, 2*H1*2*H2, H2];
disp(['Network size: ' num2str(layers)]);

%initialize network:
network = init_network(size_left,size_right,H1,H2); %will hold W and Bias vectors.

%inital error:
[a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
error = mean(log(ones(size(lv1,1),1)+exp(-lv1.*a3')));
disp(['Error before training: ' num2str(error)]);
incorrect_rate = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
disp(['Mispredicted percentage at start : ' num2str(incorrect_rate)]);

%% Backpropagation:
network_delta = init_network_delta(H1,H2,size_left,size_right);

num_batches=25;
for i=1:num_batches
    for j=1:size(dlt1,1) %send each image through pipeline once and update for each
        rand_batch = randperm(size(dlt1,1));
        data_left = dlt1(rand_batch(j),:);
        data_right = drt1(rand_batch(j),:);
        labels_left = lt1(rand_batch(j),:);
        [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(data_left, data_right, network);
        [network, network_delta] = backprop(network, labels_left, data_left, data_right, nu, pi, network_delta, a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2);
    end
    [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
    error = mean(log(ones(size(lv1,1),1)+exp(-lv1.*a3')));
    disp(['Error at batch ' num2str(i) ' : ' num2str(error)]);
    incorrect_rate = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
    disp(['Mispredicted percentage at batch ' num2str(i) ' : ' num2str(incorrect_rate)]);
end
