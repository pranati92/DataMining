% Pranati Waghmare  1001200937
% Apoorva Karkhanis 1001167312

% This function implements kNN for HandWritten characters file
function[accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, train_data, test_data)
    % getting number of rows
    
    col_size = size(train_data,2);

    % finding Unique Values
    [unique_values,~,index] = unique(train_data(1,:));  
    unique_values_count = numel(unique_values);

    % calculating remainder
    remainder = mod(col_size,unique_values_count);
    train_data = train_data(:,1:(col_size-remainder));

    % getting number of rows
    col_size = size(train_data,2);

    % calculating Rows Per Fold
    cols_per_uniq_val = col_size / unique_values_count;

    testFold = floor(cols_per_uniq_val / k_value);

    % initializing Accuracy variables
    accuracyK = int16.empty(0,k_value);
    accuracyC = int16.empty(0,k_value);
    accuracyL = int16.empty(0,k_value);
    accuracyS = int16.empty(0,k_value);

    correctCount=0;
    n = testFold;
    l = 1;
    count = 0;
    cnt = 1;

    % creating a copy for setting folds
    rows_per_fold_copy = cols_per_uniq_val;
    divided_val = ceil(rows_per_fold_copy / k_value);

    for i = 1:k_value
        if(rows_per_fold_copy > divided_val)
            fold(i) = divided_val;
            rows_per_fold_copy = rows_per_fold_copy - divided_val;
        else
            fold(i) = rows_per_fold_copy;
            if(rows_per_fold_copy - divided_val >= 0)
                rows_per_fold_copy = rows_per_fold_copy - divided_val;
            end
        end
    end

    for i = 1:k_value
        n = fold(i);
        col = [];
        test = [];
        train = [];

        for k = cnt:n:col_size
            col = [col cnt:(cnt+(n-1))];
            cnt = cnt + cols_per_uniq_val;        
        end

        col(col > col_size) = [];
        dup = train_data;
        B = dup(:,col);
        test = [test B];

        count = count + n;
        cnt = count + 1;
        l = l + 1;

        dup(:,col) = [];
        train = [train dup];

        % creation of Ytrain
        [number_of_train_data_rows,number_of_train_data_columns] = size(train);
        identity_matrix_for_Ytrain = eye(unique_values_count);
        number_of_ones = ones(1,number_of_train_data_columns/unique_values_count);
        Ytrain = kron(identity_matrix_for_Ytrain,number_of_ones);
        % end of Ytrain

        
       
        
        % calling kNN, Centroid and Linear Regression Classifiers
        [Ctest, Caccuracy] = Centroid(train(2:end,:),test(2:end,:),Ytrain,test(1,:),n, cols_per_uniq_val);
        [Ktest, Kaccuracy] = kNN(train(2:end,:),test(2:end,:),Ytrain,test(1,:),n,unique_values_count,cols_per_uniq_val);
        [Ltest, Laccuracy] = LinearRegression(train(2:end,:),test(2:end,:),Ytrain,test(1,:));

        % storing all accuracy
        accuracyK(i) = Kaccuracy;
        accuracyL(i) = Laccuracy;
        accuracyC(i) = Caccuracy;

        % SVM
        model_linear = svmtrain(train(1,:)',train(2:end,:)','-t 0');
        [predict_label, accuracy, prob_values] = svmpredict(test(1,:)',test(2:end,:)',model_linear);
        accuracyS(i) = accuracy(1,1);
     
        
   
    end
