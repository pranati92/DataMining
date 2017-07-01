% Pranati Waghmare  1001200937
% Apoorva Karkhanis 1001167312

% This function implements Centroid Classifier for HandWritten characters file
function[result, accuracy] = Centroid(train_data,test_data,Ytrain,test_labels,test_size,rows_per_uniq_val)

    % Finding Column size
    [~,col_size] = size(test_labels);

    centroid = (train_data * Ytrain') ./ (rows_per_uniq_val - test_size);
    
    % Calculating distance
    distance = pdist2(centroid',test_data','euclidean');
    [Ytest2value  result]= min(distance,[],1);

    difference = result - test_labels ;
    
    for i = 1:col_size
        if ne(difference(1,i),0)
            difference(1,i) = 1;
        end
    end
    
    % Calculating accuracy
    accuracy = ((col_size - sum(difference))/col_size ) * 100;
   
end