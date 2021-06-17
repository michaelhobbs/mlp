function [network] = init_network(size_left,size_right,H1,H2)

%initialize weights and bias:
%layer1
network{1}.W = sqrt(1/size_left)*randn(H1,size_left);
network{1}.B = zeros(H1,1);
network{2}.W = sqrt(1/size_right)*randn(H1,size_right);
network{2}.B = zeros(H1,1);

%layer2
%L -> L
network{3}.W = sqrt(1/H1)*randn(H2,H1);
network{3}.B = zeros(H2,1);
%L -> RL & R -> RL
network{4}.W = sqrt(1/(2*H1))*randn(H2,2*H1);
network{4}.B = zeros(H2,1);
%R -> R
network{5}.W = sqrt(1/H1)*randn(H2,H1);
network{5}.B = zeros(H2,1);

%layer3
network{6}.W = sqrt(1/(3*H2))*randn(H2,1);
network{6}.B = zeros(1,1);

end