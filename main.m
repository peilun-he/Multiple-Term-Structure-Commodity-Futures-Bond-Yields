clear;

% Add search path
addpath(genpath(pwd));

% Read data
dat_all = readtable("WTI_Formatted.csv");
date_all = table2array(dat_all(:, 2)); % trading days
date_all = datetime(date_all, 'InputFormat', 'dd/MM/yyyy');

index = find(date_all >= datetime(2010, 1, 1) & date_all <= datetime(2019, 12, 31)); 
dat = dat_all(index, 3: 69);
date = date_all(index);
price_daily = zeros(size(dat));
for i = 1: size(dat, 2)
    if isnumeric(dat{:, i})
        price_daily(:, i) = dat{:, i};
    elseif iscell(dat{:, i})
        price_daily(:, i) = str2double(dat{:, i});
    end
end

% Select the first observation of each month
[~, unique_index, ~] = unique(year(date) + month(date)/100); 
%maturity = [1, 6: 6: 67]/12; 
maturity = (1: 12)/12; 
price = price_daily(unique_index, maturity*12); % monthly price
date_monthly = date(unique_index);

% Factors extracted from US Tresury yields 
US = readtable("US_less1year_2Factors.csv");
US = table2array(US);
US_st1 = readtable("US_st1.csv");
US_st1 = table2array(US_st1);
US_st2 = readtable("US_st2.csv");
US_st2 = table2array(US_st2);

% US Treasury yield data
treasury = readtable("US_12Con.csv");
treasury = table2array(treasury(:, 2: 6));

% Global setups
dt = 1/12;
[n_obs, n_contract] = size(price);
n_factor = size(US, 2);
st1_start = datetime("01/12/2014", 'InputFormat', 'dd/MM/yyyy'); % start date of stress test 1
st1_end = datetime("31/12/2015", 'InputFormat', 'dd/MM/yyyy'); % end date of stress test 1
st2_start = datetime("01/12/2014", 'InputFormat', 'dd/MM/yyyy'); % start date of stress test 2

%%
clear;

% Add search path
addpath(genpath(pwd));

US = readtable("US_3factors_period4.csv");
US = table2array(US);

price = readtable("WTI4.csv");
price = table2array(price(:, 2:13));

maturity = (1: 12)/12; 
dt = 1/12;
[n_obs, n_contract] = size(price);
n_factor = size(US, 2);

%% Schwartz-Smith two-factor model
yt = log(price);

% Restrictions
lb = [1e-08, 1e-08, -Inf, 1e-08, 1e-08, -0.9999, -Inf, -Inf,...
    repelem(1e-08, n_contract)];
ub = [    3,     3,  Inf,     3,     3,  0.9999,  Inf,  Inf,...
    repelem(1, n_contract)];
Aineq = [-1, 1, repelem(0, 6+n_contract)];
bineq = 0; % Aineq * parameter <= bineq

% initial values 
par0 = [1, 0.5, 1, 0.5, 0.5, -0.5, 0.1, 0.1,...
    repelem(0.01, n_contract)];

% Estimate parameters
mdl1 = ssm(@(parameter) schwartz_smith_model(parameter, yt, maturity, dt));

options = optimoptions('fmincon','MaxFunEvals',50000,...
    'TolFun',1e-8,'TolX',1e-8,'MaxIter',1000,'Display','off');

[est_mdl1, par1, ~, ~, exit1] = estimate(mdl1, yt, par0, 'Display', 'off', ...
    'options', options, 'Univariate', true, 'lb', lb, 'ub', ub, 'Aineq', Aineq, 'bineq', bineq);

% Estimation errors
mu = inv(eye(2) * est_mdl1.A) * [ 0 ; par1(3)/par1(2)*(1-exp(-par1(2)*dt))];
F = AofT(par1, maturity)';
[~, ~, ~, ~, ~, ~, ~, deflated_yt1] = schwartz_smith_model(par1, yt, maturity, dt);
[deflated_states1, ~, output1] = filter(est_mdl1, deflated_yt1);
estimated_states1 = deflated_states1 + mu';
estimated_yt1 = estimated_states1 * est_mdl1.C' + F';
residual1 = exp(yt) - exp(estimated_yt1);

rmse1 = sqrt( mean(residual1 .^ 2) ); 
rmse1_month = sqrt( mean(residual1 .^ 2, 2) ); 

rmse1 = rmse1';

%% Two-factor functional regression model 
yt = log(price); 
factor = US;

% Restrictions
lb = [1e-08, 1e-08, -Inf, 1e-08, 1e-08, -0.9999, -Inf, -Inf,...
    repelem(1e-08, n_contract),...
    repelem(-Inf, n_contract*n_factor)];
ub = [    3,     3,  Inf,     3,     3,  0.9999,  Inf,  Inf,...
    repelem(1, n_contract),...
    repelem(Inf, n_contract*n_factor)];
Aineq = [-1, 1, repelem(0, 6+n_contract+n_contract*n_factor)];
bineq = 0; % Aineq * parameter <= bineq

% initial values 
par0 = [1, 0.5, 1, 0.5, 0.5, -0.5, 0.1, 0.1,...
    repelem(0.01, n_contract),...
    repelem(1, n_contract*n_factor)];

% Estimate parameters
mdl2 = ssm(@(parameter) functional_regression_model(parameter, yt, maturity, factor, dt));

options = optimoptions('fmincon','MaxFunEvals',50000,...
    'TolFun',1e-8,'TolX',1e-8,'MaxIter',1000,'Display','off');

[est_mdl2, par2, ~, ~, exit2] = estimate(mdl2, yt, par0, 'Display', 'off', ...
    'options', options, 'Univariate', true, 'lb', lb, 'ub', ub, 'Aineq', Aineq, 'bineq', bineq);

% Estimation errors
mu = inv(eye(2) * est_mdl2.A) * [ 0 ; par2(3)/par2(2)*(1-exp(-par2(2)*dt))];
G = zeros(n_contract, n_factor); 
G(:) = par2(9+n_contract: end);
F = AofT(par2, maturity)';
[~, ~, ~, ~, ~, ~, ~, deflated_yt2] = functional_regression_model(par2, yt, maturity, factor, dt);
[deflated_states2, ~, output2] = filter(est_mdl2, deflated_yt2);
estimated_states2 = deflated_states2 + mu';
estimated_yt2 = estimated_states2 * est_mdl2.C' + factor * G' + F';
residual2 = exp(yt) - exp(estimated_yt2);

rmse2 = sqrt( mean(residual2 .^ 2) ); 
rmse2_month = sqrt( mean(residual2 .^ 2, 2) ); 

rmse2 = rmse2';

% Covariance of filtered observations
cov_state = cat(3, output2.FilteredStatesCov);
%cov_Y = zeros(n_contract, n_contract, n_obs);
%for i = 1: n_obs
%    cov_Y(:, :, i) = est_mdl2.C * cov_state(:, :, i) * est_mdl2.C' + est_mdl2.D * est_mdl2.D';
%end

% Samples from state variables 
rng(1234);
n_sample = 1000;
sample_yt = zeros(n_obs, n_contract, n_sample);
for i = 61: n_obs
    if det(cov_state(:, :, i)) < 0 
        [e_vector, e_value] = eig(cov_state(:, :, i));
        e_value(e_value<0) = 1e-08;
        cov_state(:, :, i) = e_vector * e_value * e_vector';
    end
    sample_state = mvnrnd(estimated_states2(i, :), cov_state(:, :, i), n_sample);
    sample_yt(i, :, :) = (sample_state * est_mdl2.C' + factor(i, :)*G' + F')';
end

% Save data
%writematrix(G, "Data/Coe/coe_less5year_6factors.csv")

% Spread under different stress
%original = estimated_yt2; % if original US factors
%cov_Y_original = cov_Y;
%sample_original = sample_yt;

stress2 = estimated_yt2;  
%cov_Y_2 = cov_Y;
sample_stress_2 = sample_yt;


%% Plots
% Futures curve
figure;
surf(1:size(price,2), date_monthly, price);
set(gcf, 'Position', [100, 100, 900, 750]);
ax = gca;
ax.FontSize = 18;
xlabel('Maturity in months');
ylabel('Date');
zlabel('Futures price');
shading interp;
colorbar;

% Yield curve
figure;
surf([1,3,6,9,12], date_monthly, treasury);
set(gcf, 'Position', [100, 100, 900, 750]);
ax = gca;
ax.FontSize = 18;
xlabel('Maturity in months');
ylabel('Date');
zlabel('Yield');
shading interp;
colorbar;

% First and last Treasury yield 
figure;
hold on;
plot(date_monthly, treasury(:, 1), "r", "LineWidth", 1);
plot(date_monthly, treasury(:, end), "b", "LineWidth", 1);
set(gcf, 'Position', [100, 100, 900, 750]);
ax = gca;
ax.FontSize = 18;
xlabel("Date");
ylabel("Yield");
legend(["1-month Treasury", "12-month Treasury"]);
axes("Position", [0.7, 0.15, 0.2, 0.2]);
box on;
hold on;
plot(date_monthly(101:120), treasury(101:120, 1), "r", "LineWidth", 1);
plot(date_monthly(101:120), treasury(101:120, end), "b", "LineWidth", 1);

% Functional component of the first and last futures contracts 
fr_comp = US * G'; % functional regression component
figure;
hold on;
plot(date_monthly, fr_comp(:, 1), "r", "LineWidth", 1);
plot(date_monthly, fr_comp(:, end), "b", "LineWidth", 1);
set(gcf, 'Position', [100, 100, 900, 750]);
ax = gca;
ax.FontSize = 18;
xlabel("Date");
ylabel("Functional regression component");
legend(["1-month futures contract", "12-month futures contract"]);
axes("Position", [0.2, 0.15, 0.2, 0.2]);
box on;
hold on;
plot(date_monthly(101:120), treasury(101:120, 1), "r", "LineWidth", 1);
plot(date_monthly(101:120), treasury(101:120, end), "b", "LineWidth", 1);

%% Spread under different stress
shock = 1;
stress = stress1;
sample_stress = sample_stress_1;
%cov_Y_st = cov_Y_1;
st = "st1";

diff = exp(original) - exp(stress);
diff(1: 60, :) = 0;
mean_diff = zeros(size(diff, 1), 3); 
mean_diff(:, 1) = mean(diff(:, 1: 4), 2); 
mean_diff(:, 2) = mean(diff(:, 5: 8), 2);
mean_diff(:, 3) = mean(diff(:, 9: 12), 2); 

% Sampling from state variables
sample_diff2 = exp(sample_original) - exp(sample_stress);
sample_mean_diff2 = zeros(n_obs, 3, n_sample);
sample_mean_diff2(:, 1, :) = mean(sample_diff2(:, 1: 4, :), 2);
sample_mean_diff2(:, 2, :) = mean(sample_diff2(:, 5: 8, :), 2);
sample_mean_diff2(:, 3, :) = mean(sample_diff2(:, 9: 12, :), 2);
sample_mean_diff2 = sort(sample_mean_diff2, 3); 


% Mean difference of short-end, middle, and long-end maturity
colors = [228, 26, 28;
          55, 126, 184;
          77, 175, 74];
figure; 
set(gcf, 'Position', [100, 100, 900, 600]);
ax = gca;
ax.FontSize = 22; 
hold on;
for i = 1: 3
    plot(date_monthly, mean_diff(:, i), "Color", colors(i, :)/255, "LineWidth", 1);
end
for i = 1: 3
    plot(date_monthly, sample_mean_diff2(:, i, 25), "--", "Color", colors(i, :)/255, "LineWidth", 0.5);
    plot(date_monthly, sample_mean_diff2(:, i, 975), "--", "Color", colors(i, :)/255, "LineWidth", 0.5);
end
yline(0, "LineWidth", 0.5);
%ylim([-0.1, 0.3]);
yLimits = ylim;
if shock == 1
    xline(st1_start, "k", "LineWidth", 1.5);
    text(st1_start, yLimits(1), "Shock starts", ...
        'FontSize', 22, ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
    xline(st1_end, "k", "LineWidth", 1.5);
    text(st1_end, yLimits(1)+0.03, "Shock ends", ...
        'FontSize', 22, ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
elseif shock == 2
    xline(st2_start, "k", "LineWidth", 1.5);
    text(st2_start, yLimits(1), "Shock starts", ...
        'FontSize', 22, ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
end
xlabel("Date"); 
ylabel("Mean difference in futures price");
legend(["Short-end maturity", "Middle maturity", "Long-end maturity"], "Location", "northwest");

saveas(gcf, append("Mean_diff_in_", st, ".jpg"));

% Difference of all curves
colors = [166,206,227; 
          31,120,180;
          178,223,138;
          51,160,44;
          251,154,153;
          227,26,28;
          253,191,111;
          255,127,0;
          202,178,214;
          106,61,154;
          255,255,153;
          177,89,40];
figure;
set(gcf, 'Position', [100, 100, 900, 600]);
hold on;
for i = 1: size(diff, 2)
    plot(date_monthly, diff(:, i), "Color", colors(i, :)/255, "LineWidth", 1);
end
if shock == 1
    xline(st1_start, "k", "LineWidth", 1.5);
    text(st1_start, yLimits(1), "Shock starts", ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
    xline(st1_end, "k", "LineWidth", 1.5);
    text(st1_end, yLimits(1), "Shock ends", ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
elseif shock == 2
    xline(st2_start, "k", "LineWidth", 1.5);
    text(st2_start, yLimits(1), "Shock starts", ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
end
xlabel("Date"); 
ylabel("Difference in yields");
lgd = legend(["1 month", "3 months","6 months", "9 months", "1 year", "2 years", "3 years", "5 years", ...
    "7 years", "10 years", "20 years", "30 years"], "Location", "northwest");
title(lgd, "Time to maturity"); 




