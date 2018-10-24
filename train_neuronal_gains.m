%% Training recurrent neuronal activity through gain modulation
%
% This code trains the neuronal gains in a stability-optimised
% network so that a linear readout of the neural firing rates produces a
% desired target output. This code reproduces the main plots in Figure 1.
% One can also use this code for grouped gain modulation by setting the
% parameter 'num_groups' to the desired number of groups. This code takes
% approximately 5 minutes to run on a laptop with an Intel i7 2.3 GHz
% processor.
% 
% 'data.mat' contains:
% initial_cond: Initial condition for the neuronal activity
% initial_target: Initial target output with all gains set to 1
% novel_target: Novel target output
% W_rec: Recurrent weight matrix with a SOC architecture
% readout: Readout weights set such that the initial network output
% generates the initial target reach.
%
% Written by Jake Stroud

load('data') %data.mat contains the initial condition x_0, the intial network output with all gains set to 1 (initial_reach), the target network output (novel_reach), the SOC weight matrix (W_rec), and the readout weights (readout).

%% Initialise parameters
num_iterations = 5000;                              %Total number of training iterations to run for (set to at least to 4000 to ensure error saturation and so that the visualisation is reasonble)
NN = length(data.W_rec);                            %Number of neurons
n_exc = NN/2;                                       %Number of excitatory neurons
params.n_timepoints = length(data.initial_target);  %length of the output
gain_function = 'NL';                               %Type of gain function to use: Either nonlinear tanh 'NL', or linear 'L'.
params.initial_cond_noise = 0;                      %Amount of gaussian noise to add the the initial condition
params.over_tau = 1/200;                            %1/tau (ms^-1)
params.tfinal = 500;                                %Amount of time (ms) to run neuronal dynamics for
params.r0 = 20;                                     %Baseline firing rate
params.rmax = 100;                                  %Maximum firing rate

%% Grouped modulation
num_groups = 200; %Number of random modulatory groups. Set to 200 for neuron-specific modulation.

% Create random groupings if using groupings
group_index = repmat(1:num_groups,1,round(NN/num_groups));
if length(group_index) < NN
    group_index(length(group_index)+1:NN) = randsample(1:num_groups,length(length(group_index)+1:NN));
end
group_index = (randsample(group_index,NN))';

% Set up gain function
if strcmp(gain_function, 'L')           %Linear gain function
    params.f = 'f_linear';
    params.ff = 'f_linear';
elseif strcmp(gain_function, 'NL')      %Nonlinear gain function
    params.f = 'f_non_linear';
    params.ff = 'f_final_non_linear';
else                                    %Warning otherwise
    warning('Incorrect firing rate function flag given, using linear firing rate');
    params.f = 'f_linear';
    params.ff = 'f_linear';
end

%% Train neuronal gains using learning rule
% Initialise parameters
error = zeros(num_iterations,1);                            %Initialise vector of errors
T_ss = sum((data.novel_target - mean(data.novel_target)).^2); %Initial total sum of squares
gains = ones(NN,num_iterations);                            %Initialise matrix of neuronal gains for training

% Run neuronal dynamics to calculate initial error
dynamics = integrate_dynamics(data.W_rec, gains(:,1), params, data.initial_cond);

% Setup matrix of neuronal dynamics and offset
design = zeros(params.n_timepoints,NN/2 +1);
design(:,1) = ones(params.n_timepoints,1); %The offset bias for the readout weights
design(:,2:end) = dynamics.R(:,1:n_exc);

%Calculate initial output and error
initial_output = design*data.readout;
error(1) = sum((initial_output-data.novel_target).^2)/T_ss;

% Initialise parameters
output = zeros(params.n_timepoints,num_iterations);     %Initialise network output for each training iteration
output(:,1) = initial_output;
alpha = 0.3;                                            %Parameter used in learning rule
gains_bar = gains(:,1);                                 %Low pass filter of previous gains
error_bar = error(1);                                   %Low pass filter of previous errors
R = 0;                                                  %Modulatory signal

figure  %Create figure for plotting the error over training iterations
for iteration = 2:num_iterations
    
    xi = 0.001*randn(num_groups,1); %Gaussian noise added to neuronal gains at each trial
    
    % Learning rule update
    gains(:,iteration) = gains(:,iteration-1) + ...
        R*(gains(:,iteration-1) - ...
        gains_bar) + xi(group_index);
    
    % Run neuronal dynamics on new gains after learning rule update
    dynamics = integrate_dynamics(data.W_rec, gains(:,iteration), params, data.initial_cond);
    
    % Recalculate output and error
    design(:,2:end) = dynamics.R(:,1:n_exc);    
    output(:,iteration) = design*data.readout;
    error(iteration) = sum((output(:,iteration)-data.novel_target).^2)/T_ss;
    
    % Update modulatory signal and filtered traces of error and gains
    R = sign(error_bar - error(iteration));
    error_bar = alpha*error_bar + (1-alpha)*error(iteration);
    gains_bar = alpha*gains_bar + (1-alpha)*gains(:,iteration);    
    
    % Print current number of completed iterations and plot the error
    if mod(iteration,100) == 0
        plot(error(1:iteration))
        ylabel('Error'); xlabel('Number of iterations')
        pause(0.01)
        iteration = iteration
    end
end

% Find minimum error over all training iterations and extract the best gains and output
[~,II] = min(error);
bestgains = gains(:,II);
bestoutput = output(:,II);

%% Plotting
% Plot error reduction and 5 outputs over training
figure
subplot(2,3,4)
plot(error,'r')
ylabel('Error'); xlabel('Number of iterations'); xlim([0 num_iterations]);
box off

axes('Position',[0.17 0.2 0.17 0.25])
hold on
cm = colormap(copper(5));
c = 1;
for i = [1 round(150.^(linspace(1,log(II)/log(200),4)))]
    
    plot(output(:,i),'color',cm(c,:))    
    c = c+1;
    
end
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('Time')
ylabel('EMG')


% Plot 10 neurons' gain changes over training
subplot(2,3,5)
plot(gains(1:10,:)','color',[0.4 0.4 0.4],'linewidth',0.5)
ylabel('Gain'); xlabel('Number of iterations'); xlim([0 num_iterations]);
box off


% Plot outputs from 10 noisy initial conditions
params.initial_cond_noise = 30;
subplot(2,3,6)
box off
hold on
plot(initial_output,'k')
plot(data.novel_target,'color',(1/255)*[214 124 42])
for i = 1:10
    
    dynamics = integrate_dynamics(data.W_rec, gains(:,1), params, data.initial_cond);
    design(:,2:end) = dynamics.R(:,1:n_exc);
    plot(design*data.readout,'color',[0.4 0.4 0.4],'linewidth',0.3)
    
    dynamics = integrate_dynamics(data.W_rec, bestgains, params, data.initial_cond);
    design(:,2:end) = dynamics.R(:,1:n_exc);
    plot(design*data.readout,'color',[0.4 0.4 0.4],'linewidth',0.3)
    
end
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('Time')
ylabel('EMG')