function [A, B, C, D, mean0, cov0, state_type, DeflateY] = schwartz_smith_model(parameter, yt, maturity, dt)
% For logarithm of futures price y(t), and hidden state vector x(t), 
% the original model is given as: 
% State equation:       x(t) = E   + A   * x(t-1) + B * u(t)
% Observation equation: y(t) = F_t + C_t * x(t)   + D * e(t)
%
% The standard state-space model is given as: 
% State equation:       x(t) - mu             = A   * (x(t-1) - mu) + B * u(t)
% Observation equation: y(t) - C_t * mu - F_t = C_t * (x(t)   - mu) + D * e(t)
% where mu = (I - A)^(-1) E
%
% Inputs:
%   parameter: a vector of parameters
%   yt: logarithm of futures price
%   maturity: row vector or matrix
%   dt: delta t 

kappa      = parameter(1);
gamma      = parameter(2);
mu         = parameter(3);
sigma_chi  = parameter(4);
sigma_xi   = parameter(5);
rho        = parameter(6);
lambda_chi = parameter(7);
lambda_xi  = parameter(8);

n_obs = size(yt, 1); % yt: N*T matrix
n_contract = size(yt, 2);

A = [exp(-kappa*dt), 0; 0, exp(-gamma*dt)]; % 2*2 matrix
B11 = (1-exp(-2*kappa*dt)) / (2*kappa) * sigma_chi^2;
B12 = (1-exp(-(kappa+gamma)*dt)) / (kappa+gamma) * sigma_chi*sigma_xi*rho;
B22 = (1-exp(-2*gamma*dt)) / (2*gamma) *sigma_xi^2;
cov_mat = [B11, B12; B12, B22]; % 2*2 matrix, covariance matrix
B = chol(cov_mat)';
D = diag(parameter(9: end)); % T*T matrix

mu = inv(eye(2) * A) * [ 0 ; mu/gamma*(1-exp(-gamma*dt))]; % 2*1 matrix

if isvector(maturity)
    % fixed maturity for monthly data
    C = [exp(-kappa*maturity); exp(-gamma*maturity)]'; % T*2 matrix
    F = AofT(parameter, maturity)'; % T*1 matrix
    DeflateY = yt - (C * mu)' - F'; 
else 
    % roll down maturity for daily/weekly data
    C = cell(n_obs, 1);
    F = cell(n_obs, 1);
    DeflateY = zeros(n_obs, n_contract);
    for i = 1: n_obs 
        C(i) = { [exp(-kappa*maturity(i,:)); exp(-gamma*maturity(i,:))]' };
        F(i) = { AofT(parameter, maturity(i,:))' };
        DeflateY(i, :) = yt(i, :)' - C * mu - cell2mat(F(i));
    end
end

mean0      = [];
cov0       = [];
state_type = [];

end