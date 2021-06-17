function [d1,d2,d3] = preprocess_data(train,valid,test)
all_train = double([train]); % IDEA: treat validation data like test data -> mu only a function of pure training data
mu = mean(all_train,1);
sigma = sqrt(mean(bsxfun(@minus, all_train, mu).^2,1)); %std could be better
sigma_minus = sigma.^-1;
train = double(train);
valid = double(valid);
test = double(test);

d1 = bsxfun(@times, bsxfun(@minus, train, mu), sigma_minus);
d2 = bsxfun(@times, bsxfun(@minus, valid, mu), sigma_minus);
d3 = bsxfun(@times, bsxfun(@minus, test, mu), sigma_minus);




% IDEA: compare with std
%     bsxfun(@minus, all_train, mu); this removes row vector mu from
%     every row of matrix all_train.
%
%     bsxfun(@times, bsxfun(@minus, test, mu), sigma_minus); this
%     multiplies element-by-element each row of (test-mu) matrix by
%     sigma_minus row vector.


end