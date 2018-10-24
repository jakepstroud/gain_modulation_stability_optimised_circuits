%% Script to create and initial 'unstable' weight matrix and a stability-optimised circuit

% This code creates an initial 'unstable' weight matrix W according to
% (Hennequin et al., Neuron, 2014) and then creates the stability-optimised
% variant so that the resulting neuronal dynamics display rich activity transients.
%
% Written by Jake Stroud

% Create intial 'unstable' weight matrix
N = 200;            %Number of neurons
p = 0.1;            %Density of connections
R = 10;             %Initial approximate spectral abscissa prior to stability optimisation
gamma = 3;          %The inhibition/excitation ratio

W = initialnet(N, p, R, gamma); %Create initial 'unstable' weight matrix

% Create stabiliy-optimised circuit using the initial weight matrix W
rate = 10;              %Gradient-descent learning rate
desired_SA = 0.15;      %Ultimate desired spectral abscissa after stability optimisation

% Create stability-optimised circuit
Wsoc = soc_function(W, rate, desired_SA, gamma);