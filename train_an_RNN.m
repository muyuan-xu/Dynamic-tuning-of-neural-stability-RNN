%% Set parameters.
% Network architecture.
V = 6; % Two rule-dependent inputs, two phase-dependent inputs, and two sensory inputs.
N = 100;
M = 1;
layer_size = [V N M];
nconnections = inf;
g_by_layer = [1.0 1.0 1.0]; % Defines the initial scale of synapses for each layer.
layer_type = {'linear', 'tanh', 'linear'}; % Defines the transfer function for each layer.

% Simulation parameters.
nics = 1; % Number of initial conditions.
tau = 20.0;
dt  = 1.0;
net_noise_sigma = 0.1;

% Optimization parameters.
objfun = 'sum-of-squares';
mu = 0.03;
lambda_init = 0.0002;
maxcgiter = 100;
mincgiter = 10;
cg_tol = 1e-6;
do_learn_biases = 1;
do_random_init_biases = 1;
do_learn_init_state = 1;
do_random_init_state = 1;
% Terminate conditions.
max_hf_iter = 1000;
max_hf_fail_count = 500;
objfun_tol = 1e-5;
objfun_diff_tol = 1e-8;

% Initialize the network.
disp('Initializing network.');
net = init_rnn(layer_size, g_by_layer, layer_type, objfun, ...
    'nconnections', nconnections, 'nics', nics, 'tau', tau, 'dt', dt, 'net_noise_sigma', net_noise_sigma, 'mu', mu, 'do_learn_biases', do_learn_biases, ...
    'do_random_init_biases', do_random_init_biases, 'do_learn_init_state', do_learn_init_state, 'do_random_init_state', do_random_init_state);

%% Define the task.
% We implemented two paradigms: transient rule vs. tonic rule (see Remington et al., 2018 for detail).
T_transient   = 100; % Duration of the rule-dependent input in the transient-rule paradigm.
T_minprecue   = 200; % Minimal length of the precue period (before the switch of the phase-dependent inputs).
T_maxprecue   = 500; % Maximal length of the precue period.
T_delay       = 100; % This corresponds to the delay period.
T_sensory     = 200; % Duration of the sensory input/integration.
T_minpostintg = 0;   % Minimal length of the trial after sensory integration.
T = T_maxprecue + T_delay + T_sensory + T_minpostintg; % Total length of the trial.

ntransient    = round(T_transient/dt);
ndelay        = round(T_delay/dt);
nsensory      = round(T_sensory/dt);

times  = dt:dt:T;
ntimes = length(times);

num_of_rules = 2;
rules = 1:num_of_rules;

num_of_precue_intervals = 5; % How many different precue intervals do we use to train the network.
all_nprecue   = round(linspace(T_minprecue, T_maxprecue, num_of_precue_intervals)/dt);
all_npostintg = ntimes - all_nprecue - ndelay - nsensory;

num_of_stimuli = 2;
assert(V == 4 + num_of_stimuli, 'Error input dimension!');
M_rule    = 0.5; % Magnitude of the rule-dependent input.
M_phase   = 0.5; % Magnitude of the phase-dependent input.
M_sensory = 0.1; % Magnitude of the sensory input.
input_noise_sigma = 0.1;

ntrials = 1000;
size_of_minibatches = 1000;

init_conditions = cell(1, ntrials); % Allows one to set different initial conditions.
inputs          = cell(1, ntrials);
targets         = cell(1, ntrials);

for i = 1:ntrials
    init_conditions{i} = 1;
    inputs{i} = input_noise_sigma * sqrt(dt) * randn(4 + num_of_stimuli, ntimes);
    % Define the rule-dependent inputs.
    idx_rule = randi(num_of_rules);
    inputs{i}(idx_rule, :) = inputs{i}(idx_rule, :) + M_rule * ones(1, ntimes); % Tonic rule paradigm.
    % inputs{i}(idx_rule, 1:ntransient) = inputs{i}(idx_rule, 1:ntransient) + M_rule * ones(1, ntransient); % Transient rule paradigm.
    % Define the phase-dependent inputs.
    idx_precue = randi(num_of_precue_intervals);
    nprecue    = all_nprecue(idx_precue);
    npostintg  = all_npostintg(idx_precue);
    inputs{i}(3, :) = inputs{i}(3, :) + [M_phase * ones(1, nprecue) zeros(1, ndelay + nsensory + npostintg)];
    inputs{i}(4, :) = inputs{i}(4, :) + [zeros(1, nprecue) M_phase * ones(1, ndelay + nsensory + npostintg)];
    % Define the sensory inputs.
    idx_stimulus = randi(num_of_stimuli);
    inputs{i}(4 + idx_stimulus, :) = inputs{i}(4 + idx_stimulus, :) + [zeros(1, nprecue + ndelay) M_sensory * ones(1, nsensory) zeros(1, npostintg)];
    % Define the target.
    if idx_rule == 1
        % For rule 1, the first half stimuli are associated with outcome 1
        % and the second half are associated with outcome 2.
        if idx_stimulus <= round(num_of_stimuli/2)
            outval = 1;
        else
            outval = -1;
        end
    else
        % For rule 2, the association is reversed.
        if idx_stimulus <= round(num_of_stimuli/2)
            outval = -1;
        else
            outval = 1;
        end
    end
    targets{i} = NaN(M, ntimes);
    targets{i}(1, nprecue + ndelay) = 0;
    targets{i}(1, ntimes) = outval;
end

%% Do parallel computing.
do_parallel = true;
nworkers = 7;
if do_parallel
    mycluster = parcluster('local');
    if mycluster.NumWorkers < nworkers
        mycluster.NumWorkers = nworkers;
        saveProfile(mycluster);
    end
    if matlabpool('size') == 0
        matlabpool('local', nworkers);
    elseif matlabpool('size') ~= nworkers
        matlabpool close;
        matlabpool('local', nworkers);
    end
end

%% Save snapshots for training.
save_path = './data/';
if (~exist(save_path, 'dir'))
    mkdir(save_path);
end

save_every = 20;

%% Start training.
[theta_opt, objfun] = hfoptimizer(net, init_conditions, inputs, targets, ...
'do_parallel', do_parallel, 'save_path', save_path, 'save_every', save_every, 'size_of_minibatches', size_of_minibatches, 'lambda_init', lambda_init, ...
'min_cg_iter', mincgiter, 'max_cg_iter', maxcgiter, 'cg_tol', cg_tol, 'max_hf_iter', max_hf_iter, 'max_hf_fail_count', max_hf_fail_count, ...
'objfun_tol', objfun_tol, 'objfun_diff_tol', objfun_diff_tol);

% if do_parallel
%     matlabpool close;
% end
