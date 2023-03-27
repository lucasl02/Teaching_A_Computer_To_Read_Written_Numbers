%%% Clean workspace
clear all; close all; clc

%%% Load training data

load('CP4_training_labels.mat')
load('CP4_training_images.mat');


% Reshape training_images to 784x30000 in order to be consistent with the
% codes from the lecture.

train_images = reshape(training_images,[784 30000]);

idx = randperm(size(train_images,2));

train_images = train_images(:,idx(1:size(train_images,2)/4));

%%% Projecting onto principal component

% Conduct the wavelet transform on the entire training dataset.  To make it
% reusable you can write it as a function very similar (almost exactly the
% same) as dc_wavelet.m from the lecture.  For MATLAB you will need to
% include the function at the end of the code.

% For the transform ignore the above and use the variable Training_DWT
load('Training_DWT.mat')

% Find the SVD of the transformed data just like in Week7_LDA.m

[U,S,V] = svd(Training_DWT,'econ');

% Plot singular values (include in report, but not in gradescope submission)

%plot(diag(S),'')
%title('Singular Values of SVD')
%grid on; 

% How many features (i.e., singular values) should we use?  Save this as
% A1.  Hint: it's less than what we used in the lecture

A1 = 15; % 1x1. The number of PCA modes we are going to project onto.


% Project onto the principal components just like we did in Week7_LDA.m
% Restrict the matrix U to that of the feature space (like we did in 
% dc_trainer.m).  Save this as A2

projection_matrix = S*V'; 
A2 = U(:,1:A1); %196 x 15


%%% Pick two numbers to train/test on.  Use 0 and 1 for the autograder.

% This is going to be quite different from what we did in the lectures.  In
% the lecture we had two datasets with dogs and cats.  Here everything is
% jumbled up so we need to separate them out.  Separate all the training 
% images of 0's and 1's using the training labels.  Hint: a for loop and
% some if statements should be sufficient.

zero_matrix = []; 
one_matrix = []; 
all_labels = training_labels(idx(1:size(train_images,2)/4));



for i=1:size(all_labels,1)
    if all_labels(i) == 9
        zero_matrix = [zero_matrix projection_matrix(:,i)];
    end
    if all_labels(i) ~= 9
        one_matrix = [one_matrix projection_matrix(:,i)];
    end
end

zero_matrix = zero_matrix(1:A1,:);
one_matrix = one_matrix(1:A1,:);

% Calculate the within class and between class variances just like in 
% Week7_LDA.m.  Save these as A3 and A4.

mean_zeroes = mean(zero_matrix,2);
mean_ones = mean(one_matrix,2);

A3 = 0; % within class variances
for k = 1:size(zero_matrix,2)
    A3 = A3 + (zero_matrix(:,k) - mean_zeroes)*(zero_matrix(:,k) - mean_zeroes)';
end
for k = 1:size(one_matrix,2)
   A3 =  A3 + (one_matrix(:,k) - mean_ones)*(one_matrix(:,k) - mean_ones)';
end

A4 = (mean_zeroes - mean_ones)*(mean_zeroes-mean_ones)'; %inter class variances

%A3 = ; % 15x15
%A4 = ; % 15x15




% Find the best projection line just like in Week7_LDA.m.  Save the
% normalized projection line w as A5

[V2, D] = eig(A4,A3); % linear disciminant analysis; i.e., generalized eval. prob.
[lambda, ind] = max(abs(diag(D)));
A5 = V2(:,ind);
A5 = A5/norm(A5,2);

%A5 = ; % 15x1


% Project the training data onto w just like in Week7_LDA.m

v_ones = A5'*one_matrix;
v_zeroes = A5'*zero_matrix;
% Find the threshold value just like in Week7_LDA.m.  Save it as A6

if mean(v_ones) > mean(v_zeroes)
    A5 = -A5;
    v_ones = -v_ones;
    v_zeroes = -v_zeroes;
end

sortones = sort(v_ones);
sortzeroes = sort(v_zeroes);
t1 = length(sortones); % start on the right
t2 = 1; % start on the left
while sortones(t1) > sortzeroes(t2) 
    t1 = t1 - 1;
    t2 = t2 + 1;
end
%go past each other
A6 = (sortones(t1) + sortzeroes(t2))/2; % get the midpoint
%A6 = 2.7947; % 1x1

plot(v_ones,zeros(size(v_ones,2)),'ob','Linewidth',2)
hold on; 
plot(v_zeroes,zeros(size(v_zeroes,2)),'dr','Linewidth',2)
title('Projection of Nine onto Projection Vector');
grid on;
%





%%% Classify test data

load('CP4_test_labels.mat')
load('CP4_test_images.mat')


% Reshape test_images to 784x5000 in order to be consistent with the
% codes from the lecture.

test_images = reshape(test_images,784,5000);

% From the test set pick out the 0's and 1's without revealing the labels.
% Save only the images of 0's and 1's as a new dataset and save the
% associated labels for those exact 0's and 1's as a new vector.

% Wavelet transform:  you can just use the same function you did for the
% training portion.

% For the transform ignore the above and use the variable Test_DWT
load('Test_DWT.mat')


% Project the test data onto the principal components just like in
% Week7_Learning.m
% Save the results in a vector (just like in Week7_Learning.m) and save it
% as A7.

A7 = U'*Test_DWT; 
A7 = A7(1:15,:);
A7 = A5'*A7;
%A7 = ; % 1x1062


%%% Checking performance just like we did in Week7_Learning.m.  If you did
%%% everything like I did (which may or may not be optimal), you should
%%% have a success rate of 0.9972.

%% 
clear all; close all; clc; 

load('CP4_training_labels.mat')

load('Training_DWT.mat')

[U,S,V] = svd(Training_DWT,'econ');

projection_matrix = S*V'; 

A2 = U(:,1:15); 

all_labels = training_labels; 

%%% For report only, not for the autograder:  Now write an algorithm to
%%% classify all 10 digits.  One way to do this is by using the "one vs all
%%% " method; i.e., loop through the digits and conduct LDA on each digit
%%% vs. all the other digits.

for num = 0:1
    num_matrix = [];
    othernums_matrix = [];
    for i=1:size(all_labels,1)
        if all_labels(i) == num
            num_matrix = [num_matrix projection_matrix(:,i)];
        end
        if all_labels(i) ~= num
            othernums_matrix = [othernums_matrix projection_matrix(:,i)];
        end
    end
    
    num_matrix = num_matirx(1:15,:);
    othernums_matrix = othernums_matrix(1:15,:);

    mean_num = mean(num_matrix,2);
    mean_othernums = mean(othernums_matrix,2);

    withinclass_var = 0; % within class variances
    for k = 1:size(num_matrix,2)
        A3 = A3 + (num_matrix(:,k) - mean_num)*(num_matrix(:,k) - mean_num)';
    end
    for k = 1:size(othernums_matrix,2)
       A3 =  A3 + (othernums_matrix(:,k) - mean_othernums)*(othernums_matrix(:,k) - mean_othernums)';
    end
    
    interclass_var = (mean_othernums - mean_num)*(mean_othernums-mean_num)'; %inter class variances
   
    [V2, D] = eig(interclass_var,withinclass_var); % linear disciminant analysis; i.e., generalized eval. prob.
    [lambda, ind] = max(abs(diag(D)));
    projection_line = V2(:,ind);
    projection_line = projection_line/norm(projection_line,2);

    v_num = projection_line'*num_matrix; 
    v_othernums = projection_line'*othernums_matrix;


    plot(v_num,zeros(size(v_num)),'ob','Linewidth',2)
    hold on; 
    plot(v_othernums,size(v_othernums),'dr','Linewidth',2)
    title('Projection of Objects onto Projection Vector');
    grid on;
end


%%% Put any helper functions here
