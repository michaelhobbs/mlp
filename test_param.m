function [validation_error, train_error] = test_param(nu, pi,network,H1,H2)
load('2class.mat');

dlt1=dlt1(1:50,:);
drt1=drt1(1:50,:);
lt1=lt1(1:50,:);
% dlv1=dlv1(1:50,:);
% drv1=drv1(1:50,:);
% tdl1=tdl1(1:50,:);
% dlt1=tdr1(1:50,:);



[dlt1,dlv1,tdl1] = preprocess_data(dlt1,dlv1,tdl1);
[drt1,drv1,tdr1] = preprocess_data(drt1,drv1,tdr1);


% ********************* MLP FOR BINARY CLASSIFICATION *********************

%% CONSTANTS:
% H1 = 2;
% H2 = 4; %note this needs to change
size_left = size(dlt1,2); %we should remove one, consider both cameras both resolution
size_right = size(drt1,2);
nu_start = nu;
% pi = pi;
num_batches=10;
% num_epochs = 15; % when this is too high, we overfit the network (ex:error(25) > error(5))?
validation_error = [];
validation_incorrect_rate = [] ;
train_error = [];
train_incorrect_rate = [];

batch_size = size(lt1,1)/num_batches; % take care to make num_batches such that number of samples is a multiple of it (we could make this better by treating the last batch on its own, it would have fewer samples in it than the other batches)

% cmap = distinguishable_colors(num_epochs);
%error_epoch = zeros(num_epochs,1);
%incorrect_epoch = zeros(num_epochs,1);
%incorrect_rate = zeros(num_batches,1);

layers = [(size_left + size_right)*H1, 2*H1*2*H2, H2];
disp(['Network size: ' num2str(layers)]);

%initialize network:
%  network = init_network(size_left,size_right,H1,H2); %will hold W and Bias vectors.
 
 %inital error:
 [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
validation_error(1) = mean(log(ones(size(lv1,1),1)+exp(-lv1.*a3')));
 disp(['Error before training: ' num2str(validation_error(1))]);
 incorrect_rate(1) = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
    disp(['Mispredicted percentage at start : ' num2str(incorrect_rate(1))]);
    
     [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlt1,drt1,network);
train_error(1) = mean(log(ones(size(lt1,1),1)+exp(-lt1.*a3')));
 disp(['Error before training: ' num2str(train_error(1))]);
 train_incorrect_rate(1) = sum(sign(lt1)~=sign(a3'))/size(lt1,1)*100;
    disp(['Mispredicted percentage at start : ' num2str(train_incorrect_rate(1))]);

%% Backpropagation:
network_delta = init_network_delta(H1,H2,size_left,size_right);

k=1;
% for k=1:num_epochs % at each epoch make a random permutation of entire training set
    
while ( k==1 || validation_error(k)<validation_error(k-1) )
    %nu = nu_start*exp(-k); % exponential decay is bad
    %nu=nu_start/k;
    nu=nu;
    rand_epoch = randperm(size(dlt1,1));
     for i=1:num_batches % in each epoch send random permuted data through in batches
%          for j=1:batch_size % send each image of current batch through network
             rand_batch = rand_epoch((i-1)*batch_size+1:i*batch_size);
             data_left = dlt1(rand_batch,:);
             data_right = drt1(rand_batch,:);
             labels_left = lt1(rand_batch,:);
             [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(data_left, data_right, network);
             [network, network_delta] = backprop(network, labels_left, data_left, data_right, nu, pi, network_delta, a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2);
%          end
%          [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
%          error(i)= mean(log(ones(size(lv1,1),1)+exp(-lv1.*a3'))); 
%          disp(['Error at batch ' num2str(i) ' : ' num2str(error(i))]);
%          incorrect_rate(i) = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
%          disp(['Mispredicted percentage at batch ' num2str(i) ' : ' num2str(incorrect_rate(i))]);
%  
        
%         fig1=plot (i,error(i),'+');
%         hold on;
%         set(fig1,'xdata',k);
     end
     [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlv1,drv1,network);
%      top = -lv1.*a3';
%      meanneg = mean(log1p(top(find(top<0))));
%      meanpos = mean(log(   ones(size(find(top>=0))) +   exp(top(find(top>=0)))    ));
%      error(k+1) = 1/2 * (mean(log1p(top(find(top<0)))) + mean(log(ones(size(find(top>=0))+exp(top(find(top>=0)))))));
     validation_error(k+1)= mean(-lv1.*a3'+log(ones(size(lv1,1),1)+exp(lv1.*a3'))); 

     incorrect_rate(k+1) = sum(sign(lv1)~=sign(a3'))/size(lv1,1)*100;
     if mod(k,100)==0
        disp(['Validation error at epoch ' num2str(k) ' : ' num2str(validation_error(k+1))]);
     end
    
     [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(dlt1,drt1,network);
train_error(k+1) = mean(log(ones(size(lt1,1),1)+exp(-lt1.*a3')));
 train_incorrect_rate(k+1) = sum(sign(lt1)~=sign(a3'))/size(lt1,1)*100;

     %disp(['Validation miss rate at epoch ' num2str(k) ' : ' num2str(incorrect_rate(k+1))]);
     
% ONLY DO TEST ERROR AT END     
%      [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(tdl1,tdr1,network);
%      error_epoch(k) = mean(log(ones(size(tl1,1),1)+exp(-tl1.*a3')));
%      incorrect_epoch(k) = sum(sign(tl1)~=sign(a3'))/size(tl1,1)*100;
k=k+1;
end
% hold off;
plot([0:k-1],validation_error);
hold on;
plot([0:k-1],train_error,'r');
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