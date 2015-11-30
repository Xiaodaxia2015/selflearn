function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); %25 X401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  %10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1=[ones(m,1) X];%5000X401

z2=a1*Theta1';%5000X401*401*25
a2=[ones(m,1) sigmoid(z2)]; %5000X26

z3=a2*Theta2'; %5000X26*26*10
a3=sigmoid(z3); %5000X10

yt=zeros(m,num_labels); %5000X10
for i=1:m
    k=y(i);
    yt(i,k)=1;
end

for l=1:num_labels
    J=J+1/m*(-yt(:,l)'*log(a3(:,l))-(ones(m,1)-yt(:,l))'*log(ones(m,1)-a3(:,l)));
end
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
d3=a3-yt; %5000X10
middle2=d3*Theta2; %5000X10 * 10*26=5000X26
middle2=middle2(:,2:hidden_layer_size+1); %5000X25
d2=middle2.*sigmoidGradient(z2);%  5000X25

D1=zeros(size(Theta1)); %25X401

D2=zeros(size(Theta2)); %10X26

D1=D1+d2'*a1; % 25X401

D2=D2+d3'*a2; %10X5000*5000X26=10X26

Theta1_grad=1/m*D1;

Theta2_grad=1/m*D2;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Theta12=Theta1(:,2:input_layer_size+1).^2;
% Theta22=Theta2(:,2:hidden_layer_size+1).^2;
% J=J+(lambda/(2*m))*(sum(sum(Theta12))+sum(sum(Theta22)));

Regt1=Theta1;
Regt1(:,1)=0;
Regt2=Theta2;
Regt2(:,1)=0;
J=J+(lambda/(2*m))*(sum(sum(Regt1.^2))+sum(sum(Regt2.^2)));
Theta1_grad=Theta1_grad+lambda/m*Regt1;
Theta2_grad=Theta2_grad+lambda/m*Regt2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
