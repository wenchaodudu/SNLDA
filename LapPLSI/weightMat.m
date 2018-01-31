function W = weightMat(X, k)
% WEIGHTMAT generates the weight matrix based on nearest neighbor and
% cosine distance
[n, ~] = size(X);
disp('Beginning constructing neighborhood graph.')
idX = knnsearch(X, X, 'K', k, 'Distance', 'cosine');
disp('Finish constructing neighborhood graph.')
W = sparse(n, n);
for i = 1:n
    for j = 1:k
        W(i,idX(i,j)) = 1 - pdist(X([i,idX(i,j)],:),'cosine');
        if W(idX(i,j),i) == 0
            W(idX(i,j),i) = 1 - pdist(X([i,idX(i,j)],:),'cosine');
        end
    end
    disp(['Finish', num2str(i), 'th iteration.'])
end
end
