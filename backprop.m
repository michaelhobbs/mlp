function [network, network_delta,r2_L] = backprop(network, labels_left, X_L, X_R, nu, pi, old_network_delta, a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2)
network_delta = cell(6,1); %will hold W and Bias update vectors.
labels_right = labels_left;
%% Layer 3:
% Find residuals:
labels_L_R = [labels_left]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r3 = -labels_L_R ./ (ones(size(labels_L_R)) + exp(labels_L_R .* a3));
% Find gradients:
gradw3 = r3 * z2';
gradb3 = mean(r3,2);

%% Layer 2:
% Find residuals:
r2_L = r3'*network{6}.W' .*(a2_LR.*exp(a2_L+a2_R) ./ ( (exp(a2_L)+1).^2 .* (exp(a2_R)+1) ))';
r2_R = r3'*network{6}.W' .*(a2_LR.*exp(a2_R+a2_L) ./ ( (exp(a2_R)+1).^2 .* (exp(a2_L)+1) ))';
r2_LR = r3'*network{6}.W' .* (exp(a2_L+a2_R))' ./ ((ones(size(a2_L))+exp(a2_L)) .* (ones(size(a2_R))+exp(a2_R)))';

% Find gradients:
gradw2_L = r2_L' * z1_L';
gradw2_R = r2_R' * z1_R';
gradw2_LR = [r2_LR' * z1_L' , r2_LR' * z1_R'];
gradb2_L = mean(r2_L,1);
gradb2_R = mean(r2_R,1);
gradb2_LR = mean(r2_LR,1);

%% Layer 1:
% Find residuals:
half = 1/2 * size(network{4}.W,2);
r1_L = r2_L*network{3}.W .* (sech(a1_L)').^2 + r2_LR*network{4}.W(:,1:half) .* (sech(a1_L)').^2;
r1_R = (r2_R*network{5}.W .* (sech(a1_R)').^2 + r2_LR*network{4}.W(:,half+1:end) .* ((sech(a1_R)').^2));

% Find gradients:
gradw1_L = r1_L' * X_L;
gradw1_R = r1_R' * X_R;
gradb1_L = mean(r1_L',2);
gradb1_R = mean(r1_R',2);




% Perform updates:
network_delta{6}.W = -nu*(1-pi)*gradw3' + pi*old_network_delta{6}.W;
network_delta{6}.B = -nu*(1-pi)*gradb3' + pi*old_network_delta{6}.B;
network{6}.W = network{6}.W+network_delta{6}.W;
network{6}.B = network{6}.B+network_delta{6}.B;


% Perform updates:
network_delta{3}.W = -nu*(1-pi)*gradw2_L + pi*old_network_delta{3}.W;
network_delta{3}.B = -nu*(1-pi)*gradb2_L' + pi*old_network_delta{3}.B;
network{3}.W = network{3}.W+network_delta{3}.W;
network{3}.B = network{3}.B+network_delta{3}.B;

network_delta{5}.W = -nu*(1-pi)*gradw2_R + pi*old_network_delta{5}.W;
network_delta{5}.B = -nu*(1-pi)*gradb2_R' + pi*old_network_delta{5}.B;
network{5}.W = network{5}.W+network_delta{5}.W;
network{5}.B = network{5}.B+network_delta{5}.B;

network_delta{4}.W = -nu*(1-pi)*gradw2_LR + pi*old_network_delta{4}.W;
network_delta{4}.B = -nu*(1-pi)*gradb2_LR' + pi*old_network_delta{4}.B;
network{4}.W = network{4}.W+network_delta{4}.W;
network{4}.B = network{4}.B+network_delta{4}.B;


% Perform updates:
network_delta{1}.W = -nu*(1-pi)*gradw1_L + pi*old_network_delta{1}.W;
network_delta{1}.B = -nu*(1-pi)*gradb1_L + pi*old_network_delta{1}.B;
network{1}.W = network{1}.W+network_delta{1}.W;
network{1}.B = network{1}.B+network_delta{1}.B;

network_delta{2}.W = -nu*(1-pi)*gradw1_R + pi*old_network_delta{2}.W;
network_delta{2}.B = -nu*(1-pi)*gradb1_R + pi*old_network_delta{2}.B;
network{2}.W = network{2}.W+network_delta{2}.W;
network{2}.B = network{2}.B+network_delta{2}.B;
