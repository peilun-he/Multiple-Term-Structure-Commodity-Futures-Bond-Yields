%%
clear;

% Add search path
addpath(genpath(pwd));

US = readtable("Data/Data1/US_FPCA_2factors_period1.csv");
%US = readtable("US_2factors_period1.csv");
US = table2array(US);

USTreasury = readtable("USTreasury_period1.csv");
USTreasury = table2array(USTreasury);

price = readtable("WTI1.csv");
price = table2array(price(:, 2:13));

maturity = (1: 12)/12; 
dt = 1/12;
[n_obs, n_contract] = size(price);
%n_factor = size(US, 2);

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

[est_mdl1, par1, ~, logL1, exit1] = estimate(mdl1, yt, par0, 'Display', 'off', ...
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
[AIC1, BIC1] = aicbic(logL1, numel(par1), size(yt,1));

%% Two-factor functional regression model 
yt = log(price); 
factor = US;
n_factor = size(factor, 2);

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

[est_mdl2, par2, ~, logL2, exit2] = estimate(mdl2, yt, par0, 'Display', 'off', ...
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
[AIC2, BIC2] = aicbic(logL2, numel(par2), size(yt,1));

%% ARX-LSC Model 
yt = log(price); 
dim = size(yt, 2);

level = mean(USTreasury, 2);
slope = USTreasury(:, 5) - USTreasury(:, 1);
curvature = 2 * USTreasury(:, 3) - USTreasury(:, 1) - USTreasury(:, 5);
X = [level, slope, curvature];

rmse = zeros(dim, 1); 
AIC = zeros(dim, 1);
BIC = zeros(dim, 1); 

for i = 1: size(yt, 2)
    Y = yt(2:end, i);
    Z = [yt(1:(end-1), i), X(2:end, :)]; 

    est_ARX_LSC = fitlm(Z, Y);

    fitted_log = predict(est_ARX_LSC, Z);
    fitted_price = exp(fitted_log);

    actual_price = price(2:end, i);

    rmse(i) = sqrt(mean((actual_price - fitted_price).^2));
    AIC(i) = est_ARX_LSC.ModelCriterion.AIC;
    BIC(i) = est_ARX_LSC.ModelCriterion.BIC;
end

rmse_all = mean(rmse);
AIC_all = sum(AIC);
BIC_all = sum(BIC);

%% F-test for non-linearity
yt = log(price); 
dim = size(yt, 2);

X = [US(:, 1), US(:, 2)];
q = 3; % number of additional non-linear coefficients 

F_stat = zeros(dim, 1); 
p_value = zeros(dim, 1);

for i = 1: size(yt, 2)
    Y = yt(2:end, i);
    Z1 = [yt(1:(end-1), i), X(2:end, :)]; 
    Z2 = [yt(1:(end-1), i), X(2:end, :), X(2:end, :).^2, X(2:end, 1).*X(2:end, 2)]; 

    mdl_re = fitlm(Z1, Y); % restricted model
    mdl_un = fitlm(Z2, Y); % unrestricted model

    SSE_re = mdl_re.SSE;
    SSE_un = mdl_un.SSE;
    
    F_stat(i) = ( (SSE_re - SSE_un)/q ) / ( SSE_un/mdl_un.DFE ); 
    p_value(i) = fcdf(F_stat(i), q, mdl_un.DFE, 'upper');
end

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



