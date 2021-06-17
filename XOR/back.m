function [network, delta] = back(XOR,labels,network,a1,a2,z1,eta)
r2 = -labels ./ bsxfun(@plus, exp(labels.*a2'),1);
delta{2}.W = -eta*r2*z1;
delta{2}.B = -eta*r2;
network{2}.W = network{2}.W+delta{2}.W;
network{2}.B = network{2}.B+delta{2}.B;

r1 = r2 .* network{2}.W .* sech(a1);
delta{1}.W = -eta*r1*XOR;
delta{1}.B = -eta*r1;
network{1}.W = network{1}.W+delta{1}.W;
network{1}.B = network{1}.B+delta{1}.B;

end