% Pranati Waghmare  1001200937
% Apoorva Karkhanis 1001167312

clc
clearvars

% %when 1st row has class labels
% %data_file = xlsread('nba.xlsx');
data_file = load('handwritten.txt');
% class_labels = data_file(1,:);
% data = data_file(2:end,:);

%test_data for row label
test_data = load('handwritten.txt');
%

% % %when last column has class labels
 %x=xlsread('nba_train.xlsx');
% x= load('iristextdata.txt');
 %data_file = rot90(x);

 % Test Data for column labels
 %test1 = load('iristextdata.txt');
  %  test1 = xlsread('nba_test.xlsx');
   % test_data = rot90(test1);


    class_labels = data_file(1,:);
    data = data_file(2:end,:);
    
% finding Unique Values
[unique_values,~,index] = unique(data_file(end,:));
number_of_classes = numel(unique_values);

[number_of_rows,number_of_columns] = size(data_file);

group_mean = mean(data,2);

% calculating mean of all group means
overall_mean = mean(group_mean);

root_mean_square = ((group_mean - overall_mean).^2).*number_of_columns;

features = number_of_rows-1;

rms_per_feature = root_mean_square./features;

sw = zeros(number_of_rows-1,1);
for i=1:(number_of_rows-1)
    sw(i,1) = sum((data(i,1:end)-group_mean(i,1)).^2); 
end

fw = (number_of_columns-1)*number_of_classes;

MSw = sw./fw;

F=rms_per_feature./MSw;

[B,I]= sort(F, 'descend');

sorted_data = data(I,:);

sorted_data = vertcat(class_labels,sorted_data);

num_of_features_labels=[100];

avg_accuracyK = [];
avg_accuracyL = [];
avg_accuracyC = [];
avg_accuracyS = [];

%odd: r(1:2:end)    even: r(2:2:end)

for num_of_features=number_of_rows;  
    train_data = sorted_data(:,:);
    
     k_value = 5;
     
      [accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, train_data, test_data);
    
    % calculating average accuracy
    avg_accuracyK = [avg_accuracyK mean(accuracyK)]
    avg_accuracyL = [avg_accuracyL mean(accuracyL)]
    avg_accuracyC = [avg_accuracyC mean(accuracyC)]
    avg_accuracyS = [avg_accuracyS mean(accuracyS)]
end

hold on;
hline1 = refline([0 mean(accuracyK)]);
hline1.Color = 'r';
hline2 = refline([0 mean(accuracyL)]);
hline2.Color = 'b';
hline3 = refline([0 mean(accuracyC)]);
hline3.Color = 'g';
hline4 = refline([0 mean(accuracyS)]);
hline4.Color = 'c';
hold off;

label1 = 'kNN';
label2 = 'Linear Regression';
label3 = 'Centroid';
label4 = 'SVM';
legend([hline1;hline2;hline3;hline4], label1,label2,label3,label4);

% displaying results
disp('===================================================');
disp('===================================================');
disp('Results : -')

disp('Accuracy for kNN               :');
disp(avg_accuracyK);

disp('Accuracy for Linear Regression :');
disp(avg_accuracyL);

disp('Accuracy for Centroid          :');
disp(avg_accuracyC);

disp('Accuracy for SVM               :');
disp(avg_accuracyS);


disp('===================================================');
disp('===================================================');