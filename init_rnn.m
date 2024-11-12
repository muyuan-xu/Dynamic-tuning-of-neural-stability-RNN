function net = init_rnn(layer_size, g_by_layer, layer_type, objfun, varargin)
% layer_size = [dim_in dim_rnn dim_out]. Input, recurrent, and output
% layer are denoted by 'u', 'r', and 'z', respectively. The corresponding
% weight matrices are denoted by Wru, Wrr, and Wzr.
nlayers = 3;
assert(length(layer_size) == nlayers, 'layer_size should be [dim_in dim_rnn dim_out]');
assert(length(layer_type) == nlayers, 'layer_type should define the transfer function for each layer.');
assert(length(g_by_layer) == nlayers, 'g_by_layer should define the initial scale of synapses for each layer.');
dim_in  = layer_size(1);
dim_rnn = layer_size(2);
dim_out = layer_size(3);
transfun_params = []; % Additional parameters for transfer functions.
nconnections = Inf; % Sparse connections can be implemented by changing this value.
net.nlayers = nlayers;
net.g_by_layer = g_by_layer;

nics = 1; % The number of initial conditions.
tau = 1.0; % By default, the dynamics is discrete, i.e. dt = tau.
dt  = 1.0; % Note that dt should never be greater than tau.
net_noise_sigma = 0.0; % The std of noise. At each step, this value will be multiplied by sqrt(dt) (for discrete model, this is 1).

mu = 1.0; % The structral damping parameter.
do_learn_init_state = true;
do_random_init_state = true;
do_learn_biases = true;
do_random_init_biases = true;
bias_scale = 1.0;

optargin = size(varargin,2);
for i = 1:2:optargin
    switch varargin{i}
        case 'n_Wru_init_v'
            n_Wru_init_v = varargin{i+1};
        case 'n_Wrr_init_n'
            n_Wrr_init_n = varargin{i+1};
        case 'm_Wzr_init_n'
            m_Wzr_init_n = varargin{i+1};
        case 'transfun_params'
            transfun_params = varargin{i+1};
        case 'nconnections'
            nconnections = varargin{i+1};
        case 'nics'
            nics = varargin{i+1};
        case 'n_x0_c'
            n_x0_init_c = varargin{i+1};
        case 'tau'
            tau = varargin{i+1};
        case 'dt'
            dt = varargin{i+1};
        case 'net_noise_sigma'
            net_noise_sigma = varargin{i+1};
        case 'mu'
            mu = varargin{i+1};
        case 'do_learn_init_state'
            do_learn_init_state = varargin{i+1};
        case 'do_random_init_state'
            do_random_init_state = varargin{i+1};
        case 'do_learn_biases'
            do_learn_biases = varargin{i+1};
        case 'do_random_init_biases'
            do_random_init_biases = varargin{i+1};
        case 'bias_scale'
            bias_scale = varargin{i+1};
        case 'n_Wru_entry_mask_v'
            n_Wru_entry_mask_v = varargin{i+1};
        case 'n_Wrr_entry_mask_n'
            n_Wrr_entry_mask_n = varargin{i+1};
        case 'm_Wzr_entry_mask_n'
            m_Wzr_entry_mask_n = varargin{i+1};
        case 'n_x0_entry_mask_c'
            n_x0_entry_mask_c = varargin{i+1};
        case 'n_bx_entry_mask_1'
            n_bx_entry_mask_1 = varargin{i+1};
        case 'm_bz_entry_mask_1'
            m_bz_entry_mask_1 = varargin{i+1};
        case 'n_Wru_cost_mask_v'
            n_Wru_cost_mask_v = varargin{i+1};
        case 'n_Wrr_cost_mask_n'
            n_Wrr_cost_mask_n = varargin{i+1};
        case 'm_Wzr_cost_mask_n'
            m_Wzr_cost_mask_n = varargin{i+1};
        case 'n_x0_cost_mask_c'
            n_x0_cost_mask_c = varargin{i+1};
        case 'n_bx_cost_mask_1'
            n_bx_cost_mask_1 = varargin{i+1};
        case 'm_bz_cost_mask_1'
            m_bz_cost_mask_1 = varargin{i+1};
        otherwise
            assert(false,['Don''t recognize ' varargin{i} '.']);
    end
end

% Initialize the weight matrices: W{1} = Wru, W{2} = Wrr, and W{3} = Wzr.
for layer = 1:nlayers
    if layer == 1
        npre = dim_in;
        npost = dim_rnn;
    elseif layer == 2
        npre = dim_rnn;
        npost = dim_rnn;
    else
        npre = dim_rnn;
        npost = dim_out;
    end
    
    W{layer} = zeros(npost,npre);
    for i = 1:npost
        if isinf(nconnections)
            idxs = [1:npre];
            W{layer}(i,idxs) = randn(1,npre);
        else
            idxs = randperm(npre);
            idxs(nconnections+1:end) = [];
            W{layer}(i,idxs) = randn(1,nconnections);
        end
        n = norm(W{layer}(i,:));
        W{layer}(i,idxs) = W{layer}(i,idxs) / n * g_by_layer(layer);
    end
    if layer == 2
        D = eig(W{layer});
        if dt/tau < 1.0
            n = max(max(real(D))); % Use spectral abscissa for continuous-time models.
        else
            n = max(max(abs(D))); % Use spectral radius for discrete-time models.
        end
        W{layer} = W{layer} / n * g_by_layer(layer);
    end
    layers(layer).npre = npre;
    layers(layer).npost = npost;
    layers(layer).type = layer_type{layer};
    
    if strcmpi(layers(layer).type,'logistic')
        layers(layer).transfun = @(x) 1.0 ./ (1.0 + exp(-x));
        layers(layer).invfun = @(y) log(y ./ (1-y));
        layers(layer).Doperator = @(y) y.*(1.0-y);
        layers(layer).D2operator = @(y) assert(false,'Function not implemented yet.');
    elseif strcmpi(layers(layer).type,'linear')
        layers(layer).transfun = @(x) x;
        layers(layer).invfun = @(y) y;
        layers(layer).Doperator = @(y) ones(size(y));
        layers(layer).D2operator = @(y) zeros(size(y));
    elseif strcmpi(layers(layer).type,'exp')
        layers(layer).transfun = @exp;
        layers(layer).invfun = @log;
        layers(layer).Doperator = @(y) y;
        layers(layer).D2operator = @(y) y;
    elseif strcmpi(layers(layer).type,'rectlinear')
        layers(layer).transfun = @(x)  (x > 0) .* x;
        layers(layer).invfun = @(y) assert(false,'No inverse for rectified linear function.');
        layers(layer).Doperator = @(y) (y > 0) .* ones(size(y));
        layers(layer).D2operator = @(y) zeros(size(y));
    elseif strcmpi(layers(layer).type,'tanh')
        layers(layer).transfun = @tanh;
        layers(layer).invfun = @atanh;
        layers(layer).Doperator = @(y) 1.0-y.^2;
        layers(layer).D2operator = @(y) -2.0 * y .* (1.0-y.^2);
        layers(layer).Dinvfun = @(y) 1.0 ./ (1.0 - y.^2);
    elseif strcmpi(layers(layer).type,'recttanh')
        layers(layer).transfun = @(x) (x > 0) .* tanh(x);
        layers(layer).invfun = @(y) assert(false,'No inverse for rectified tanh function.');
        layers(layer).Doperator = @(y) (y > 0) .* (1.0 - y.^2);
        layers(layer).D2operator = @(y) (y > 0) .* (-2.0 * y .* (1.0-y.^2));
    elseif strcmpi(layers(layer).type,'stanh')
        alpha = transfun_params(1);
        layers(layer).transfun = @(x) (1.0/alpha) * tanh(alpha*x);
        layers(layer).invfun = @(y) (1.0/alpha) * atanh(alpha*y);
        layers(layer).Doperator = @(y) 1.0-(alpha*y).^2;
        layers(layer).D2operator = @(y) assert(false,'Function not implemented yet.');
        layers(layer).Dinvfun = @(y) assert(false,'Function not implemented yet.');
    elseif strcmpi(layers(layer).type,'LHK') % Larry, Haim, Kanaka model.
        R0 = transfun_params(1);
        layers(layer).transfun = @(x)  (R0.*tanh(x./R0) .* (x <= 0.0)) + ((2.0-R0).*tanh(x./(2.0-R0)) .* (x > 0.0));
        layers(layer).invfun = @(y) assert(false,'Function not implemented yet.');
        layers(layer).Doperator = @(y)  ((1 - y.^2./R0.^2) .* (y <= 0.0)) + ((1 - y.^2./(2.0-R0).^2) .* (y > 0.0));
        layers(layer).D2operator = @(y) assert(false,'Function not implemented yet.');
        layers(layer).Dinvfun = @(y) assert(false,'Function not implemented yet.');
    elseif strcmpi(layers(layer).type,'rectstanh')
        alpha = transfun_params(1);
        layers(layer).transfun = @(x) (x > 0) .* ((1.0/alpha) * tanh(alpha*x));
        layers(layer).invfun = @(y) assert(false,'No inverse for rectified stanh function.');
        layers(layer).Doperator = @(y) (y > 0) .* (1.0-(alpha*y).^2);
        layers(layer).D2operator = @(y) assert(false,'Function not implemented yet.');
        layers(layer).Dinvfun = @(y) 1.0 ./ (1.0 - y.^2);       
    else
        assert(false,['Don''t recognize ' layers(layer).type '.']);
    end
    
    bias{layer} = zeros(npost,1); % Only bias{2} and bias{3} are used.
    if do_random_init_biases
        bias{layer} = bias_scale * 2.0*(rand(npost,1)-0.5);
    end
    
    if layer == 1
        layers(layer).nparams = npre * npost + nics * npost; % The number of weights + nics * biases.
    else
        layers(layer).nparams = npre * npost + npost; % The number of weights + biases.
    end
end
if exist('n_Wru_init_v','var')
    assert(size(n_Wru_init_v,1) == dim_rnn, 'Inconsistent layer_size and n_Wru_init_v.');
    assert(size(n_Wru_init_v,2) == dim_in, 'Inconsistent layer_size and n_Wru_init_v.');
    W{1} = n_Wru_init_v;
end
if exist('n_Wrr_init_n','var')
    assert(size(n_Wrr_init_n,1) == dim_rnn, 'Inconsistent layer_size and n_Wrr_init_n.');
    assert(size(n_Wrr_init_n,2) == dim_rnn, 'Inconsistent layer_size and n_Wrr_init_n.');
    W{2} = n_Wrr_init_n;
end
if exist('m_Wzr_init_n','var')
    assert(size(m_Wzr_init_n,1) == dim_out, 'Inconsistent layer_size and m_Wzr_init_n.');
    assert(size(m_Wzr_init_n,2) == dim_rnn, 'Inconsistent layer_size and m_Wzr_init_n.');
    W{3} = m_Wzr_init_n;
end
net.layers = layers;

% Set the initial state of the RNN.
if exist('n_x0_init_c','var')
    assert(size(n_x0_init_c,1) == dim_rnn, 'Inconsistent layer_size and n_x0_init_c.');
    assert(size(n_x0_init_c,2) == nics, 'Inconsistent nics and n_x0_init_c.');
else
    n_x0_init_c = zeros(dim_rnn,nics);
    if do_random_init_state
        n_x0_init_c = 2.0*(rand(dim_rnn,nics)-0.5);
    end
end

net.nics = nics;
assert(dt/tau <= 1.0, 'dt should not be greater than tau.');
net.tau = tau;
net.dt = dt;
net.noise_sigma = net_noise_sigma * sqrt(net.dt);

theta = packRNN(net, W{1}, W{2}, W{3}, n_x0_init_c, bias{2}, bias{3});
net.theta = theta;
net.theta_init = net.theta;

if strcmpi(objfun,'cross-entropy')
    net.objfun = 'cross-entropy';
elseif strcmpi(objfun,'sum-of-squares')
    net.objfun = 'sum-of-squares';
elseif strcmpi(objfun,'nll-poisson')
    net.objfun = 'nll-poisson';
else
    assert(false, 'Objective function not implemented yet.');
end
net.iscanonical = false;
if strcmp(net.objfun,'cross-entropy') && (strcmp(net.layers(end).type,'logistic') || strcmp(net.layers(end).type,'softmax'))
    net.iscanonical = true;
elseif strcmp(net.objfun,'sum-of-squares') && strcmp(net.layers(end).type,'linear')
    net.iscanonical = true;
elseif strcmp(net.objfun,'nll-poisson') && strcmp(net.layers(end).type,'exp')
    net.iscanonical = true;
end

net.mu = mu;
net.do_learn_biases = do_learn_biases;
net.do_learn_init_state = do_learn_init_state;

% Set the modifiable mask and cost mask.
% By default, all parameters are modifiable.
if ~exist('n_Wru_entry_mask_v','var')
    n_Wru_entry_mask_v = ones(dim_rnn, dim_in);
end
if ~exist('n_Wrr_entry_mask_n','var')
    n_Wrr_entry_mask_n = ones(dim_rnn, dim_rnn);
end
if ~exist('m_Wzr_entry_mask_n','var')
    m_Wzr_entry_mask_n = ones(dim_out, dim_rnn);
end
if ~exist('n_x0_entry_mask_c','var')
    n_x0_entry_mask_c = ones(dim_rnn, nics);
end
if ~exist('n_bx_entry_mask_1','var')
    n_bx_entry_mask_1 = ones(dim_rnn, 1);
end
if ~exist('m_bz_entry_mask_1','var')
    m_bz_entry_mask_1 = ones(dim_out, 1);
end
modifiable_mask = packRNN(net, n_Wru_entry_mask_v, n_Wrr_entry_mask_n, m_Wzr_entry_mask_n, n_x0_entry_mask_c, n_bx_entry_mask_1, m_bz_entry_mask_1);
% By default, all parameters have no cost associated with them.
if ~exist('n_Wru_cost_mask_v','var')
    n_Wru_cost_mask_v = zeros(dim_rnn, dim_in);
end
if ~exist('n_Wrr_cost_mask_n','var')
    n_Wrr_cost_mask_n = zeros(dim_rnn, dim_rnn);
end
if ~exist('m_Wzr_cost_mask_n','var')
    m_Wzr_cost_mask_n = zeros(dim_out, dim_rnn);
end
if ~exist('n_x0_cost_mask_c','var')
    n_x0_cost_mask_c = zeros(dim_rnn, nics);
end
if ~exist('n_bx_cost_mask_1','var')
    n_bx_cost_mask_1 = zeros(dim_rnn, 1);
end
if ~exist('m_bz_cost_mask_1','var')
    m_bz_cost_mask_1 = zeros(dim_out, 1);
end
cost_mask = packRNN(net, n_Wru_cost_mask_v, n_Wrr_cost_mask_n, m_Wzr_cost_mask_n, n_x0_cost_mask_c, n_bx_cost_mask_1, m_bz_cost_mask_1);
% Should check that the costs for unmodifiable parameters are zero.
net.modifiable_mask = modifiable_mask;
net.cost_mask = cost_mask;

end