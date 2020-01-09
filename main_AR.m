clear all
clc
close all
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
addpath(genpath('D:\0 学习 ☆☆☆\博士期间工作 ☆☆☆\常用数据库'));  

load randomprojection_AR.mat; 
DATA = DATA./ repmat(sqrt(sum(DATA .* DATA)), [size(DATA, 1) 1]); %normalize
c = length(unique(Label));

%% select training and test samples
train_num = 14;
Train_num = [0, repmat(train_num, 1, c)];
for ii = 1 : 10
train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    randindex = index(randperm(length(index)));
    train_data = [train_data DATA(:,randindex(1 : train_num))];
    train_label = [train_label  Label(randindex(1 : train_num))];
  
    test_data = [test_data DATA(:, randindex(train_num + 1 : end))];
    test_label = [test_label  Label(randindex(train_num + 1 : end))];
end
    
for i = 1 : size(train_data, 2)
    a = train_label(i);
    H_train(a, i) = 1;
end 

%% parameters
alpha = 1e0;
beta = 1e-2;
lambda = 1e0;
tic
[Q, S, value_AR] = FDLSR(train_data, c, H_train, Train_num, alpha, beta, lambda);
train_time(ii) = toc;
%% classfication
T_train = Q * train_data;
T_test = Q * test_data; 
T_train = T_train./ repmat(sqrt(sum(T_train .* T_train)), [size(T_train, 1) 1]);
T_test = T_test./ repmat(sqrt(sum(T_test .* T_test)), [size(T_test, 1) 1]);

mdl = fitcknn(T_train', train_label);
tic
class_test = predict(mdl, T_test');
acc(ii) = sum(test_label' == class_test)/length(test_label)*100 
test_time(ii) = toc;
ii = ii + 1;
end

% end
imshow(Q * train_data)


mean(acc)
std(acc)



plot(value_AR)


