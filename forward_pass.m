function [a1_L,a1_R,a2_L,a2_LR,a2_R,a3, z1_L, z1_R, z2] = forward_pass(data_left, data_right, network)% we also have to return hidden values because they are needed in backpropagation

%% Forward pass: (will probably be run on batches of 100 data samples)

% Layer 1:
a1_L = bsxfun(@plus,network{1}.W * data_left', network{1}.B);
a1_R = bsxfun(@plus,network{2}.W * data_right', network{2}.B);

z1_L = arrayfun(@tanh,a1_L);
z1_R = arrayfun(@tanh,a1_R);


% Layer 2:
a2_L = bsxfun(@plus,network{3}.W * z1_L ,network{3}.B);
a2_LR = bsxfun(@plus,network{4}.W * [z1_L; z1_R], network{4}.B);
a2_R = bsxfun(@plus,network{5}.W * z1_R, network{5}.B);


% BEWARE OF NUMERICAL ERRORS (handout page 9)
z2 = a2_LR .* (exp(a2_L+a2_R))./ ((1+exp(a2_L)) .* (1+exp(a2_R)));

% Layer 3:
a3 = bsxfun(@plus,network{6}.W' * z2 , network{6}.B);
end