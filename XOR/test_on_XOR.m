% test on XOR
eta=0.02;
XOR = [0,0;0,1;1,0;1,1];
labels = [-1;1;1;-1];

h1 = 4;
nu = 0.01;

network = cell(2,1);
network{1}.W = randn(h1,2);
network{1}.B = zeros(h1,1);

network{2}.W = randn(h1,1);
network{2}.B = 0;

[a1,a2,z1] = forward(XOR,network);

error_init = mean(log(ones(size(labels,1),1)+exp(-labels.*a2')));
disp(['Initial error : ' num2str(error_init)]);

num_batches = 1000;
for i=1:num_batches
    for j=1:size(XOR,1)
        [a1,a2,z1] = forward(XOR(j,:),network);
        [network] = back(XOR(j,:),labels(j,:),network,a1,a2,z1,eta);
    end
        [a1,a2,z1] = forward(XOR,network);
        error = mean(log(ones(size(labels,1),1)+exp(-labels.*a2')));
        disp(['Error at batch ' num2str(i) ' : ' num2str(error)]);
end