%% PNRS Validation

% Set up paths
base_dir = 'H:\untitled\MATLAB\';                       
path_imaging_reference = [base_dir,'fconn.mat'];
path_demographics_Table_reference = [base_dir,'demographcis_Table.csv'];
path_dictionary_demographics_Table_reference = [base_dir,'Dictionary_for_demographics_Table.csv'];
path_group_Design_Table_reference = [base_dir,'Group_Design_Table.csv'];
output_folder_BWAS = [base_dir,'BWAS_age_prediction'];  % Changed output folder name
path_parcellation_table = [base_dir,'Parcel_hcp.mat'];
path_Group_Color_Table_reference = [base_dir,'Group_Color_Table.csv'];


model = 'age ~ brain_feature-1';

% Run BWAS analysis
run_BWAS(path_imaging_reference,...
    path_demographics_Table_reference,...
    path_dictionary_demographics_Table_reference,...
    path_group_Design_Table_reference,...
    'output_folder', output_folder_BWAS,...
    'model', model,...
    'path_parcellation_table', path_parcellation_table,...
    'path_Group_Color_Table', path_Group_Color_Table_reference)  % Added color table

%% Load the brain features file
brain_features = readtable('H:\untitled\MATLAB\BWAS_age_prediction\tables\brain_feature.csv');

%Extract beta weights and p-values
beta_weights = brain_features.Estimate;  % Beta weights (Column C)
p_values = brain_features.pValue;       % P-values (Column G)

% Display basic info
fprintf('Number of connections: %d\n', length(beta_weights));
fprintf('Range of beta weights: [%.4f, %.4f]\n', min(beta_weights), max(beta_weights));
fprintf('Range of p-values: [%.4f, %.4f]\n', min(p_values), max(p_values));

%Save as separate variables
%save('beta_weights.mat', 'beta_weights');
%save('p_values.mat', 'p_values');

%Or save together
bwas_results.beta = beta_weights;
bwas_results.pval = p_values;
bwas_results.tstat = brain_features.tStat;  % Also save t-statistics if needed
%save('bwas_extracted_results.mat', 'bwas_results');

%Create a simple summary
fprintf('\nSummary Statistics:\n');
fprintf('Significant connections (p < 0.05): %d\n', sum(p_values < 0.05));
fprintf('Mean absolute beta weight: %.4f\n', mean(abs(beta_weights)));
fprintf('Strongest positive beta: %.4f\n', max(beta_weights));
fprintf('Strongest negative beta: %.4f\n', min(beta_weights));

%%
%PNRS = fconn * beta_weights; % fconn is the vectorized connectivity matrix
PNRS = connectivity_data * beta_weights; % fconn is the vectorized connectivity matrix
[r, p_value] = corr(PNRS, behavioral_data);

fprintf('\nCorrelation Results:\n');
fprintf('Correlation (r) = %.4f\n', r);
fprintf('P-value = %.4e\n', p_value);

plot_results(PNRS, behavioral_data, 'PNRS Results')

%% Plot function

function plot_results(predictions, actual, titleStr)
    % Plot prediction results with scatter plot and error distribution.
    
    if nargin < 3
        titleStr = '';
    end
    
    if isempty(predictions)
        return;
    end
    
    % Set publication-quality parameters
    figure('Position', [100, 100, 1000, 400]);
    
    subplot(1, 2, 1);

    % Plot the scatter points - MATLAB will auto-scale
    plot(actual, predictions, 'o', ...
         'MarkerSize', 5, ...
         'MarkerFaceColor', [0.267, 0.447, 0.769], ...
         'MarkerEdgeColor', 'k', ...
         'LineWidth', 0.5);
    hold on;

    % No need to restrict min/max - let MATLAB auto-scale
    min_val_x = min(actual);
    max_val_x = max(actual);

    % Calculate correlation and statistics
    [r_val, p_val] = corr(actual(:), predictions(:));

    % Calculate regression line from correlation coefficient
    mean_actual = mean(actual);
    mean_pred = mean(predictions);
    std_actual = std(actual);
    std_pred = std(predictions);

    % Regression line slope and intercept from correlation
    slope = r_val * (std_pred / std_actual);
    intercept = mean_pred - slope * mean_actual;

    % Plot the regression line across the actual data range
    x_line = [min_val_x, max_val_x];
    y_line = slope * x_line + intercept;
    plot(x_line, y_line, 'r-', 'LineWidth', 2);
    
    xlabel('Actual Values', 'FontSize', 11, 'FontWeight', 'normal');
    ylabel('Predicted Values', 'FontSize', 11, 'FontWeight', 'normal');
    
    % Format p-value for display
    if p_val < 0.001
        p_text = 'p < 0.001';
    else
        p_text = sprintf('p = %.3f', p_val);
    end
    
    % Add text (removed R²)
    text(0.05, 0.95, sprintf('r = %.3f\n%s', r_val, p_text), ...
         'Units', 'normalized', 'FontSize', 9, 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'white', 'EdgeColor', 'none');
    
    grid on;
    set(gca, 'GridAlpha', 0.3, 'GridLineStyle', '-', 'LineWidth', 0.5);
    set(gca, 'Layer', 'bottom');
    
    % Error distribution
    subplot(1, 2, 2);
    
    errors = predictions - actual;
    h = histogram(errors, 20, 'Normalization', 'pdf', ...
                  'EdgeColor', 'k', 'LineWidth', 0.5, ...
                  'FaceColor', [0.439, 0.678, 0.278], 'FaceAlpha', 0.8);
    hold on;
    
    plot([0, 0], ylim, 'k--', 'LineWidth', 1.5, 'Color', [0, 0, 0, 0.8]);
    
    mu = mean(errors);
    sigma = std(errors);
    xmin = min(xlim);
    xmax = max(xlim);
    x = linspace(xmin, xmax, 100);
    y = normpdf(x, mu, sigma);
    plot(x, y, 'k-', 'LineWidth', 1.5, 'Color', [0, 0, 0, 0.8]);
    
    mae = mean(abs(errors));
    rmse = sqrt(mean(errors.^2));
    
    xlabel('Prediction Error', 'FontSize', 11, 'FontWeight', 'normal');
    ylabel('Density', 'FontSize', 11, 'FontWeight', 'normal');
    
    text(0.95, 0.95, sprintf('MAE = %.3f\nRMSE = %.3f', mae, rmse), ...
         'Units', 'normalized', 'FontSize', 9, ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
         'BackgroundColor', 'white', 'EdgeColor', 'none');
    
    % Grid for y-axis only
    set(gca, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.3, 'GridLineStyle', '-', 'LineWidth', 0.5);
    set(gca, 'Layer', 'bottom');
    
    % Remove top and right spines for both subplots
    ax1 = subplot(1, 2, 1);
    ax2 = subplot(1, 2, 2);
    for ax = [ax1, ax2]
        set(ax, 'Box', 'off');
        set(ax, 'TickDir', 'out', 'TickLength', [0.02, 0.025], 'LineWidth', 0.8);
    end
    
    % Add panel labels
    subplot(1, 2, 1);
    text(-0.1, 1.05, 'A', 'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');
    subplot(1, 2, 2);
    text(-0.1, 1.05, 'B', 'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Set font
    set(findall(gcf, '-property', 'FontName'), 'FontName', 'Arial');
    
    % Add title if provided
    %if ~isempty(titleStr)
    %    sgtitle(titleStr, 'y', 1.02);
    %end
end