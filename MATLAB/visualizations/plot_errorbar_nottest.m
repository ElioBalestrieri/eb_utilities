function plot_errorbar_nottest(matrix)

% errorbars with scatterplots to show single datapoints

figure; hold on;

avg_ = mean(matrix);
err_ = std(matrix)/sqrt(size(matrix,1));

for iVect = 1:size(matrix, 2)
    
    this_vect = matrix(:,iVect);
    xsc_ = (randn(numel(this_vect),1))/30 +iVect;
    scatter(xsc_, this_vect, 20, [100 192 238]/255, 'filled')
      
end

errorbar(1:length(avg_), avg_, err_, '.k', 'LineWidth', 2); 

set(gca,'XTick', 1:iVect, 'XTickLabel', 1:iVect)
ylim([min(matrix(:))-.3, max(matrix(:))+.3])

end