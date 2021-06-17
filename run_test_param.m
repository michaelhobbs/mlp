
nu = [0.5,0.3,0.1,0.01,0.001]; % 4 wins
pi = [0.7,0.5,0.2,0.1,0.01]; % 5 wins
H1 = 2;
H2 = 4;
errors = cell(size(nu,1),size(pi,1));
network = init_network(size_left,size_right,H1,H2); %will hold W and Bias vectors.
for i=1:size(nu,2)
    for j=1:size(pi,2)
    [errors{i,j}.valid, errors{i,j}.train] = test_param(nu(i), pi(j),network,H1,H2);
    end
end
    
for i=1:size(nu,2)
    figure;
    plot(errors{i,1}.valid,'r');
    hold on;
    plot(errors{i,1}.train);
    hold off;
end
% 
plot(errors{6,5}.valid(:,1:300),'r'); %compare eta 0.01 & 0.001 on first 300 epochs -> 0.01 converges a lot faster on first 50!
figure;plot(errors{5,5}.valid(:,1:300),'r');

figure;
plot(errors{5,4}.valid(:,1:300),'r'); % compare mu -> speed affected but not slope -> might as well use 0.5?
hold on;plot(errors{5,3}.valid(:,1:300));

min=10000;
mini=-1;
minj=-1;
for i=1:size(nu,2)
    for j=1:size(pi,2)
        if min>errors{i,j}.valid(size(errors{i,j}.valid,2))
            min=errors{i,j}.valid(size(errors{i,j}.valid,2));
            mini=i;
            minj=j;
        end
    end
end
    