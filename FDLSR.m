function [Q, S, value] = FDLSR(X, c, H, train_num, alpha, beta, lambda) 


[d, n] = size(X);

%% initialize
Q = H * X' * inv(X * X' + 1e-4 * eye(d));
% Q = zeros(c, d);
S = zeros(c, n);
B = 2 * H - ones(c, n);
T = H + B .* S;
maxIter = 30;
value = 0;
temp = X' * inv(X * X' + beta * eye(d));
%% starting iterations
iter = 0;
while iter < maxIter
   iter = iter + 1; 
    
   Tk = T;
   Qk = Q;
   Sk = S;
  
   %update T
   M_hat = [];
   for i = 1 : c
       Tki = Tk(:,  sum(train_num(1:i)) + 1 : sum(train_num(1:i+1)));
       M_hat = [M_hat repmat(sum(Tki, 2) / train_num(i+1), 1, train_num(i+1))];
   end
   M = repmat(sum(Tk, 2) / n, 1, n);
   T = (1 + alpha + 2 * lambda) \ (Qk * X + alpha * (H + B .* Sk) - lambda * M + lambda * 2 * M_hat);
   
   %update Q
   Q = T * temp;
   
   
   
   
   %update S
   S = max((T - H) .* B, 0);
%    
%    
% % %    
%    value1 = norm(Q * X - T, 'fro')^2;
%    value2 = norm(T - H - B .* S, 'fro')^2;
%    value3 = norm(Q, 'fro') ^ 2;
%    value4 = 0;
%    for i = 1 : c
%        Ti = T(:, sum(train_num(1:i)) + 1 : sum(train_num(1:i+1)));
%        Mi = repmat(sum(Ti, 2) / train_num(i+1), 1, train_num(i+1));   
%        M = repmat(sum(T, 2) / n, 1, train_num(i+1));
%        value4 = value4 + (norm(Ti - Mi,'fro')^2 - norm(Mi - M,'fro')^2);
%    end
%    value4 = value4 + norm(T, 'fro')^2;
%    value(iter) = value1 +  alpha * value2 + beta * value3 + 0.5 * lambda * value4; %objective function value
   
end
