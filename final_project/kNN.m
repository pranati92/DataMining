% Pranati Waghmare  1001200937
% Apoorva Karkhanis 1001167312

% This function implements kNN for HandWritten characters file
function[result, accuracy] = kNN(trainData,testData,Ytrain,test_labels,testSize,unique_values_count,rows_per_uniq_val)

    [~, test_col] = size(test_labels);
    [~, train_col] = size(Ytrain);

    remaining = rows_per_uniq_val-testSize;
    number_of_char = unique_values_count;

    train_labels = repmat(1:number_of_char,[remaining 1]);
    train_labels = train_labels(:)';

    % Creating empty Distnaces set    
    distance_vector = zeros(test_col,train_col);

    for i = 1:test_col
        for j = 1:train_col
            distance_vector(i,j) = norm(testData(:,i) - trainData(:,j));
        end
    end

    [sorted_matrix,sorted_indexes] = sort(distance_vector');
    
    % Taking transpose
    sorted_matrix = sorted_matrix';
    sorted_indexes = sorted_indexes';
    
    % Creating empty Votes set
    vote = zeros(test_col,3);

    for i = 1:test_col
        for j = 1:3
            vote(i,j) = train_labels(1,sorted_indexes(i,j));
        end
    end 
    
    final_classes = mode(vote,2);
    
    % Taking transpose    
    final_classes = final_classes';
          
    result = final_classes;
    difference = final_classes - test_labels ;
    
    for i = 1:test_col
        if ne(difference(1,i),0)
            difference(1,i) = 1;
        end
    end
    
    % Calculating accuracy
    accuracy = ((test_col - sum(difference))/test_col ) * 100;
   
end