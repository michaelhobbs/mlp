function network_delta = init_network_delta(H1,H2,size_left,size_right)
network_delta = cell(6,1); %will hold W and Bias vectors.

%initialize weights and bias:
%layer1
network_delta{1}.W = zeros(H1,size_left);
network_delta{1}.B = zeros(H1,1);
network_delta{2}.W = zeros(H1,size_right);
network_delta{2}.B = zeros(H1,1);

%layer2
%L -> L
network_delta{3}.W = zeros(H2,H1);
network_delta{3}.B = zeros(H2,1);
%L -> RL & R -> RL
network_delta{4}.W = zeros(H2,2*H1);
network_delta{4}.B = zeros(H2,1);
%R -> R
network_delta{5}.W = zeros(H2,H1);
network_delta{5}.B = zeros(H2,1);

%layer3
network_delta{6}.W = zeros(H2,1);
network_delta{6}.B = zeros(1,1);


end