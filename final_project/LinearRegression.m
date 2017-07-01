% Pranati Waghmare  1001200937
% Apoorva Karkhanis 1001167312

% This function implements Linear Regression Classifier for HandWritten characters file
function[result, accuracy] = LinearRegression(train_data,test_data,Ytrain,test_labels)

    B = pinv(train_data') * Ytrain';
    Ytest1 = B' * test_data;

    [Ytest2value  result]= max(Ytest1,[],1);
    [test_rows, test_col] = size(test_labels);

    difference = result - test_labels ;
    
    for i = 1:test_col
        if ne(difference(1,i),0)
            difference(1,i) = 1;
        end
    end
    
    % Summing all difference
    sum(difference);
    
    %Calculating Accuracy
    accuracy = ((test_col - sum(difference))/test_col ) * 100;
  
end