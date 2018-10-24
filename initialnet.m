function W = initialnet(N, p, R, gamma)

% Create random matrix where the left columns are excitatory neurons and 
% right columns are inhibitory neurons with zeros on the diagonal and 
% density p elsewhere.

NN = round(p*N*(N-1));
fill = [ones(1,NN), zeros(1,N*(N-1) - NN)];
fill = reshape(fill(randperm(N*(N-1))),N,N-1); %fill is an N by N-1 matrix with NN entries that are 1

W1 = zeros(N);
W1(1:end-1,2:end) = fill(1:end-1,:);

W2 = zeros(N);
W2(2:end,1:end-1) = fill(2:end,:);

W = triu(W1,1) + tril(W2,-1); %The off-diagonal elements of the matrix W are taken from the matrix fill 

%Create synaptic strengths as in Hennequin et al., Neuron, 2014.
w0 = sqrt(2)*R/(sqrt(p*(1-p)*(1 + gamma^2)));

W = W*(w0/sqrt(N)); %Excitatory synapses
W(:,(N/2 + 1):end) = -gamma*W(:,(N/2 +1):end); %Inhibitory synapses