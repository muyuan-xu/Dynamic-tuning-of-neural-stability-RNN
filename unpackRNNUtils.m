function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNNUtils(net, varargin)
% This function returns masks in the format of the corresponding training
% parameters.
optargin = size(varargin,2);

do_theta = false;
do_entry_mask = false;
do_cost_mask = false;

for i = 1:2:optargin
    switch varargin{i}
        case 'do_theta'
            do_theta = varargin{i+1};
        case 'do_entry_mask'
            do_entry_mask = varargin{i+1};
        case 'do_cost_mask'
            do_cost_mask = varargin{i+1};
        otherwise
            assert(false,['Variable argument ' varargin{i} ' not recognized.']);
    end
end

if ~do_theta && ~do_cost_mask && ~do_entry_mask
    do_theta = true;
end

nparams = sum([net.layers.nparams]);
nparams_cumsum = cumsum([net.layers.nparams]);
nparams_cumsum = [0 nparams_cumsum];

nlayers = net.nlayers;
nics = net.nics;
W = cell(1, nlayers);
b = cell(1, nlayers);
for i = 1:3
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);
    
    npost = net.layers(i).npost;
    npre = net.layers(i).npre;
    
    if do_theta
        W_and_b = net.theta(layer_start_idx:layer_stop_idx);
    elseif do_cost_mask
        W_and_b = net.cost_mask(layer_start_idx:layer_stop_idx);
    elseif do_entry_mask
        W_and_b = net.modifiable_mask(layer_start_idx:layer_stop_idx);
    end
    
    if i == 1
        W{i} = reshape(W_and_b(1:end-nics*npost), npost, npre);
        b{i} = reshape(W_and_b(end-nics*npost+1:end), npost, nics);
    else
        W{i} = reshape(W_and_b(1:end-npost), npost, npre);
        b{i} = W_and_b(end-npost+1:end);
    end
end

n_Wru_v = W{1};
n_Wrr_n = W{2};
n_x0_c = b{1};
n_bx_1 = b{2};
m_Wzr_n = W{3};
m_bz_1 = b{3};

end