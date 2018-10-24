function [output] = integrate_dynamics(W, gains, params, initial_cond)

%% Function that integrates the equation governing the neuronal dynamics
% Written by Jake Stroud

params.gains = gains;

% Add noise to the initial condition if indicated
if params.initial_cond_noise ~= 0
    initial_cond = awgn(initial_cond(:,1),params.initial_cond_noise,'measured');
end

% Solve the ODEs governing the neuronal dynamics
[output.t, X] = ode45(@rate_dynamics_ode, ...
    linspace(0, params.tfinal, params.n_timepoints), initial_cond, [], W, params);

% Convert neuronal activities into firing rates
output.R = feval(params.ff, X, params);

output.param = params; %Output parameters if needed

end

function x_dot = rate_dynamics_ode(t, X, W, param)

% Vector equation governing the neuronal activity
x_dot = param.over_tau*(-X + W*feval(param.f, X, param));

end

% Gain functions
function out = f_non_linear(X, param)

out = zeros(size(X));
I = X < 0;
out(I) = param.r0*tanh(param.gains(I).*X(I)/param.r0);
I2 = logical(1 - I);
out(I2) = (param.rmax-param.r0)*tanh(param.gains(I2).*X(I2)/(param.rmax-param.r0));

end

function out = f_final_non_linear(X, param)

param.gains = repmat((param.gains)', length(X),1); %Here we pass all the neuronal activities over times through the gain function so we need to repmat all the gain values
out = zeros(size(X));
I = X < 0;
out(I) = (param.r0*tanh(param.gains(I).*X(I)/param.r0));
I2 = logical(1 - I);
out(I2) = (param.rmax-param.r0)*tanh(param.gains(I2).*X(I2)/(param.rmax-param.r0));

end

function out = f_linear(X, param)

out = param.gains.*X;

end