function varargout = eval_objfun(net, init_condition, v_u_t, m_target_t, forward_pass)
%% Setup.
% Network structure.
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);
[V,T] = size(v_u_t);
[M,N] = size(m_Wzr_n);

% Transfer functions.
rec_transfun = net.layers(2).transfun;
out_transfun = net.layers(3).transfun;
rec_Doperator = net.layers(2).Doperator;

% The initial state of the RNN.
n_x0_1 = n_x0_c(:,init_condition);
% dt_o_tau.
dt_o_tau = net.dt / net.tau;
% The magnitude of noise.
noise_sigma = 0.0;
if isfield(net,'noise_sigma')
    noise_sigma = net.noise_sigma;
end

n_r0_1 = zeros(N,1);
n_dr0_1 = zeros(N,1);
n_r0_1 = rec_transfun(n_x0_1);
n_dr0_1 = rec_Doperator(n_r0_1);

if net.do_learn_init_state
    init_state_increase_t = 1;
else
    init_state_increase_t = 0;
end

% Regularizers.
do_l2_regularizer = false;
if isfield(net,'l2_regularizer')
    l2_reg_weight = net.l2_regularizer.weight;
    if l2_reg_weight > 0.0
        do_l2_regularizer = true;
        % Allow 'parameters' that are NOT modifiable by using masks. Note
        % that only modifiable parameter can have a cost.        
        modifiable_mask = net.modifiable_mask;      
        cost_mask = net.cost_mask;
    end
end

do_frobenius_norm_regularizer = false;
if isfield(net,'frobenius_norm_regularizer')
    frob_reg_weight = net.frobenius_norm_regularizer.weight;
    if frob_reg_weight > 0.0 
        do_frobenius_norm_regularizer = true;
    end
end

do_firing_rate_mean_regularizer = false;
if isfield(net,'firing_rate_mean_regularizer')
    fr_mean_reg_weight = net.firing_rate_mean_regularizer.weight;
    if fr_mean_reg_weight > 0.0
        do_firing_rate_mean_regularizer = true;
        fr_mean_reg_dv = net.firing_rate_mean_regularizer.desired_value;
        fr_mean_reg_mask = logical(net.firing_rate_mean_regularizer.mask);
        num_fr_mean_reg = length(find(fr_mean_reg_mask));
        assert(size(fr_mean_reg_mask,1) == N, 'Invalid mask.');
        assert(num_fr_mean_reg > 0, 'Invalid mask.');
    end
end

do_firing_rate_var_regularizer = false;
if isfield(net,'firing_rate_var_regularizer')
    fr_var_reg_weight = net.firing_rate_var_regularizer.weight;
    if fr_var_reg_weight > 0.0
        do_firing_rate_var_regularizer = true;
        fr_var_reg_dv = net.firing_rate_var_regularizer.desired_value;
        fr_var_reg_mask = logical(net.firing_rate_var_regularizer.mask);
        num_fr_var_reg = length(find(fr_var_reg_mask));
        assert(size(fr_var_reg_mask,1) == N, 'Invalid mask.');
        assert(num_fr_var_reg > 0, 'Invalid mask.');
    end
end

do_firing_rate_covar_regularizer = false;
if isfield(net,'firing_rate_covar_regularizer')
    fr_covar_reg_weight = net.firing_rate_covar_regularizer.weight;
    if fr_covar_reg_weight > 0.0
        do_firing_rate_covar_regularizer = true;
        fr_covar_reg_dv = net.firing_rate_covar_regularizer.desired_value;
        fr_covar_reg_mask = logical(net.firing_rate_covar_regularizer.mask);
        num_fr_covar_reg = length(find(fr_covar_reg_mask));
        assert(size(fr_covar_reg_mask,1) == N, 'Invalid mask.');
        assert(num_fr_covar_reg > 0, 'Invalid mask.');
    end
end

do_norm_pres_regularizer = false;
if isfield(net,'norm_pres_regularizer')
    norm_pres_reg_weight = net.norm_pres_regularizer.weight;
    if norm_pres_reg_weight > 0.0
        do_norm_pres_regularizer = true;
        norm_pres_reg_dv = net.norm_pres_regularizer.desired_value;
        norm_pres_reg_mask = logical(net.norm_pres_regularizer.mask);
        num_norm_pres_reg = length(find(norm_pres_reg_mask));
        assert(size(norm_pres_reg_mask,1) == N, 'Invalid mask.');
        assert(num_norm_pres_reg > 0, 'Invalid mask.');
    end
end

%% Do forward pass.
if isempty(forward_pass)
    n_x_1 = n_x0_1;
    n_r_1 = n_r0_1;

    n_x_t = zeros(N,T);
    n_r_t = zeros(N,T);
    n_Wu_t = n_Wru_v * v_u_t;
    
    n_noise_t = zeros(N,T);
    if noise_sigma > 0.0
        n_noise_t = noise_sigma * randn(N,T);
    end
    
    for t = 1:T
        n_x_1 = (1.0-dt_o_tau)*n_x_1 + dt_o_tau*(n_Wrr_n*n_r_1 + n_Wu_t(:,t) + n_bx_1 + n_noise_t(:,t));
        n_r_1 = rec_transfun(n_x_1);
        n_x_t(:,t) = n_x_1;
        n_r_t(:,t) = n_r_1;
    end
    n_dr_t = zeros(N,T);
    n_dr_t = rec_Doperator(n_r_t);
    
    m_z_t = out_transfun(m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T));
else
    n_r_t = forward_pass{1};
    n_dr_t = zeros(N,T);
    n_dr_t = rec_Doperator(n_r_t);
       
    m_z_t = out_transfun(m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T));
end

%% Calculate the objective function.
% Count the number of available target values.
vmask = ~isnan(m_target_t);
ntargets = length(find(vmask));

mtv_target_1 = m_target_t(vmask);
mtv_z_1 = m_z_t(vmask);

all_Ls = [];
switch net.objfun
    case 'cross-entropy'
        switch net.layers(end).type
            case 'cross-entropy'
            case 'logistic'
                L_output = -sum(sum(mtv_target_1 .* log(mtv_z_1+realmin) + (1 - mtv_target_1).*log(1-mtv_z_1+realmin)));
            case 'softmax'
                L_output = -sum(sum(mtv_target_1 .* log(mtv_z_1+realmin)));
            otherwise
                assert(false, 'Transfer function not supported.');
        end
    case 'sum-of-squares'
        L_output = (1.0/2.0) * sum(sum((mtv_target_1 - mtv_z_1).^2));
    case 'nll-poisson'
        assert(false, 'Objective function nll-poisson is not implemented yet.');
    otherwise
        assert(false, ['Objective function ' net.objfun ' is not implemented yet.']);
end    
L_output = L_output / ntargets; % If this normalization is changed, so should the calculation of gradient.
all_Ls(end+1) = L_output;

L_l2 = 0;
if do_l2_regularizer
    L_l2 = (l2_reg_weight/2.0)*sum((modifiable_mask .* cost_mask .* net.theta).^2);
end
all_Ls(end+1) = L_l2;

% Modified version of Frobenius norm regularization. See Sussillo et al., 2015.
L_frob = 0;
if do_frobenius_norm_regularizer
    if isfield(net.frobenius_norm_regularizer,'frob_row_idxs') % This allows to choose specific rows & columns.
        frob_row_idxs = net.frobenius_norm_regularizer.frob_row_idxs;
        frob_col_idxs = net.frobenius_norm_regularizer.frob_col_idxs;
    else
        frob_row_idxs = 1:N;
        frob_col_idxs = 1:N;
    end
    fr_Wrr_frob_fc = n_Wrr_n(frob_row_idxs,frob_col_idxs);
    fc_dr_frob_tp1 = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
    
    sW2_fc = sum(fr_Wrr_frob_fc.^2,1); % sum_i w_ij^2
    sdr2_fc = sum(fc_dr_frob_tp1.^2,2)'; % sum_t r'_j(t)^2
    s_sW2_sdr2 = sum(sW2_fc .* sdr2_fc);
    % L_frob = weight/(2T) sum_t sum_ij w_ij^2 * r'_j(t)^2.
    L_frob = frob_reg_weight/(2.0*(T+init_state_increase_t)) * s_sW2_sdr2;
end
all_Ls(end+1) = L_frob;

L_fr_mean = 0;
if do_firing_rate_mean_regularizer
    n_r_avg_1 = mean(n_r_t,2);
    % L_fr_mean = weight/(2N) sum_i (1/T sum_t r_i(t) - dv)^2.
    L_fr_mean = (fr_mean_reg_weight/(2.0*num_fr_mean_reg)) * sum((fr_mean_reg_mask .* (n_r_avg_1 - fr_mean_reg_dv)).^2);
end
all_Ls(end+1) = L_fr_mean;

L_fr_var = 0;
if do_firing_rate_var_regularizer
    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_r_var_1 = (1/T)*sum((n_r_t - repmat(n_r_avg_1, 1, T)).^2, 2);
    % L_fr_var = weight/(2N) sum_i (1/T sum_t rma_i(t)^2 - dv)^2.
    L_fr_var = (fr_var_reg_weight/(2.0*num_fr_var_reg)) * sum((fr_var_reg_mask .* (n_r_var_1 - fr_var_reg_dv)).^2);
end
all_Ls(end+1) = L_fr_var;
        
L_fr_covar = 0;
if do_firing_rate_covar_regularizer
    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_rma_t = repmat(fr_covar_reg_mask, 1, T) .* (n_r_t - repmat(n_r_avg_1, 1, T));
    n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
    n_dvI_n = (fr_covar_reg_dv * eye(N)) .* repmat(fr_covar_reg_mask, 1, N);
    % L_fr_covar = weight/(2N^2) * sum_ij (cov_ij - dv * delta_ij)^2.
    L_fr_covar = (fr_covar_reg_weight/(2.0*num_fr_covar_reg^2)) * sum(sum((n_rcov_n - n_dvI_n^2).^2));
end
all_Ls(end+1) = L_fr_covar;

% Normalize the presynaptic weights.
L_norm_pres = 0;
if do_norm_pres_regularizer
    npr_norm_pres_1 = vecnorm(n_Wrr_n(norm_pres_reg_mask,:)')';
    npr_norm2_pres_1 = npr_norm_pres_1.^2;
    % L_norm_pres = weight/(2N) * sum_i (sum_j w_ij^2 - dv^2)^2.
    L_norm_pres = (norm_pres_reg_weight/(2.0*num_norm_pres_reg)) * sum((npr_norm2_pres_1 - norm_pres_reg_dv^2).^2);
end
all_Ls(end+1) = L_norm_pres;

L = sum(all_Ls);

%% Return.
varargout = {};
varargout{end+1} = L;
varargout{end+1} = all_Ls;
varargout = {varargout};

end