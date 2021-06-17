function [a1,a2,z1] = forward(XOR,network)

a1 = bsxfun(@plus,network{1}.W * XOR', network{1}.B);
z1 = arrayfun(@tanh,a1);
a2 = bsxfun(@plus,network{2}.W' * z1, network{2}.B);