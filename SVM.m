load fisheriris

X = meas(1:100, 1:2); 
Y = species(1:100); 

SVMModel = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true);

d = 0.01;
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(SVMModel, xGrid);

figure;
gscatter(X(:,1), X(:,2), Y, 'rb', 'xo');
hold on;
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [-1 1], 'k--');
title('SVM Karar Sınırı');
xlabel('Sepal Length');
ylabel('Sepal Width');
legend('Setosa', 'Versicolor', 'Karar Sınırı');
hold off; 