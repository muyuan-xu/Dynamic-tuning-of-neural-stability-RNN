function varargout = eval_gv(net, condition, v_u_t, m_target_t, v, lambda, forward_pass)
%% Setup.
% Network structure.
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);
[V,T] = size(v_u_t);
[M,N] = size(m_Wzr_n);

% Transfer functions.
rec_transfun = net.layers(2).transfun;
out_transfun = net.layers(3).transfun;
rec_Doperator = net.layers(2).Doperator;
out_Doperator = net.layers(3).Doperator;
rec_D2operator = net.layers(2).D2operator;

% The initial state of the RNN.
n_x0_1 = n_x0_c(:,condition);
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

mu = net.mu;

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

%% Hv: Compute the Gauss-Newton forward and backward pass.
%
% The network is formally written as a mapping from the parameter space to
% the loss (see Schraudolph, 2002):
% f = L(M(N(w))), where 
% y = N(w): R^p -> R^m denotes the (linear) output,
% z = M(y): R^m -> R^m denotes the output nonlinearity,
% f = L(z): R^m -> R denotes the loss.
%
% In this case, the exact Hessian is:
% H = dJ_(L.M.N) / dw = J_N' H_(L.M) J_N + sum_i [J_(L.M)]_i H_(N_i)
% and the Gauss-Newton matrix
% G = J_N' H_(L.M) J_N is its approximation.
%
% f0 is the ordinary forward pass of a neural network, evaluating the
% function f it implements by propagating activity forward through f.
%
% r1 is the ordinary backward pass of a neural network, calculating J_f' u
% by propagating the vector u backward through f. This pass uses
% intermediate results computed in the f0 pass.
%
% f1 is based on R_v(f(w)) = J_f v, for some vector v. By pushing R_v,
% which obeys the usual rules for differential operators, down in the the
% equations of the forward pass f0, one obtains an efficient procedure to
% calculate J_f v.
%
% r2, when the R_v operator is applied to the r1 pass for a scalar function
% f, one obtains an efficient procedure for calculating Hv = R_v(J_f').
% This pass uses intermediate results from the f0, f1, and r1 passes.
%
% This is from Schraudolph, Table 1.
% pass     f0    r1(v)     f1(v)    r2
% result   f    J_f' v     J_f v    H_f v
%
% The gradient g = J'_(L.M.N) is computed by an f0 pass through the entire
% model (N, M, then L), followed by an r1 pass propagating u = 1 back
% through the entire model (L, M the N). For macthing loss functions, there
% is a shortcut since J'_(L.M) = Az+b (this is used as the definition of
% matching), we can limit the forward pass to N and M (to compute z) then
% r1-propagate u = Az+b back through just N. Note that for standard loss
% functions (linear output + lsqr error and softmax/logistic output +
% cross-entropy error) A = I and b = -z*.
%
% Hessian: after After f1-propagating v forward through N ,M, and L,
% r2-propagate R_v{1} = 0 back through the entire model (N, M, then L) to
% obtain Hv = R_v{g} (Pearlmutter, 1994). For matching loss functions, the
% shortcut is to f1-propagate v through just N and M to obtain R_v{z}, then
% r2-propagate R_v{J'_(L.M)} = A R_v{z} back through just N.
%
% Gauss-Newton: following the f1 pass, r2-propagate R_v{1} = 0 back through
% L and M to obtain R_v{J'_(L.M)} = H_(L.M) J_N v, then r1-propagate that
% back through N, giving Gv. For matching loss functions, since G = J'_N
% H_(L.M) J_N = J'_N J'_M A' J_N, we can limit the f1 pass to N, multiply
% the result with A', then r1-propagate it back through M and N. Therefore,
% we do not require an r2 pass. Alternatively, one may compute the
% equivalent Gv = J'_N A J_M J_N v by continuing the f1 pass through M,
% multiplying with A, then r1-propagating back through N only.
%
% For linear with lsqr error,                  H_(L.M) = J_M = I
% For logistic function w/ cross-entroy loss,  H_(L.M) = diag(diag(z)(1-z))
% For softmax w/ cross-entropy loss,           H_(L.M) = diag(z) - zz'
assert(net.iscanonical, 'Only canonical functions are supported.');

n_r_t = forward_pass{1};
n_dr_t = rec_Doperator(n_r_t);
m_z_t = out_transfun(m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T));

% Count the number of available target values.
vmask = ~isnan(m_target_t);
ntargets = length(find(vmask));

[n_VWru_v, n_VWrr_n, m_VWzr_n, n_vx0_c, n_vbx_1, m_vbz_1] = unpackRNN(net, v);
[n_Wru_entry_mask_v, n_Wrr_entry_mask_n, m_Wzr_entry_mask_n, n_x0_entry_mask_c, n_bx_entry_mask_1, m_bz_entry_mask_1] = unpackRNNUtils(net, 'do_entry_mask', true);

n_VWru_v = n_Wru_entry_mask_v .* n_VWru_v;
n_VWrr_n = n_Wrr_entry_mask_n .* n_VWrr_n;
m_VWzr_n = m_Wzr_entry_mask_n .* m_VWzr_n;

n_vx0_1 = n_vx0_c(:,condition);
% If we learn the initial state, then it's a parameter, so R{x(0)} = v.
% Else, if the initial condition is a constant, R{x(0)} = 0.
if net.do_learn_init_state
    n_Rx0_1 = n_x0_entry_mask_c(:,condition) .* n_vx0_1;
else
    n_Rx0_1 = zeros(N,1);
end

n_vbx_1 = n_bx_entry_mask_1 .* n_vbx_1;
m_vbz_1 = m_bz_entry_mask_1 .* m_vbz_1;
if net.do_learn_biases
    n_Rbx_1 = n_bx_entry_mask_1 .* n_vbx_1;
    m_Rbz_1 = m_bz_entry_mask_1 .* m_vbz_1;
else
    n_Rbx_1 = zeros(N,1);
    m_Rbz_1 = zeros(M,1);        
end

n_Rr0_1 = n_dr0_1 .* n_Rx0_1;
n_Rx_1 = n_Rx0_1;
n_Rr_1 = n_Rr0_1;

% Forward pass R operation up to linear output to calculate J_N * v = R{y}.
n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];
n_VWrrrm1_t = n_VWrr_n * n_rm1_t;
n_VWruu_t = n_VWru_v * v_u_t;
m_VWzrr_t = m_VWzr_n * n_r_t;

n_Rx_t = zeros(N,T);
n_Rr_t = zeros(N,T);

for t = 1:T
    n_Rx_1 = (1.0-dt_o_tau) * n_Rx_1 + dt_o_tau * (n_VWrrrm1_t(:,t) + n_Wrr_n*n_Rr_1 + n_VWruu_t(:,t) + n_Rbx_1);
    n_Rr_1 = n_dr_t(:,t) .* n_Rx_1;

    n_Rx_t(:,t) = n_Rx_1;
    n_Rr_t(:,t) = n_Rr_1;
end
m_Ry_t = repmat(m_Rbz_1,1,T) + m_VWzrr_t + m_Wzr_n*n_Rr_t;
m_dz_t = out_Doperator(m_z_t);
m_Rz_t = m_dz_t .* m_Ry_t;

% Multiply by A. For standard loss functions, A = I.

% Take transpose to exchange pre and postsynaptic neurons.
if dt_o_tau < 1.0
    n_Wrrt_dec_n = dt_o_tau * n_Wrr_n';
    n_xdec_n = (1.0-dt_o_tau);
else
    n_Wrrt_dec_n = n_Wrr_n';
    n_xdec_n = 0.0;
end
n_Wrzt_m = m_Wzr_n';

% Prepare to back pass H_(L.M) J_N v through N.
m_RdLdy_t = zeros(M,T);
if net.iscanonical
    m_RdLdy_t(vmask) = m_Rz_t(vmask);
else
    assert(false, 'R{dLdy} not implemented for noncanonical link functions.');
end
m_RdLdy_t = m_RdLdy_t / ntargets;

n_RdLextra_tp1 = zeros(N,T+1);
if do_frobenius_norm_regularizer
    % To estimate the regularization term, expand it at zero to the second
    % order and use that as an approximation. Similar to the structral
    % damping term, the zeroth and first order terms vanish since the
    % regularizer is some kind of distance. To approximate the Hessian,
    % note that there are six terms in the expansion. Specifically, L_frob
    % = weight/(2T) sum_t sum_ij w_ij^2 * r'_j(t)^2. So by the chain rule:
    % (theta is the corresponding column vector) H_kl =
    % d2L_frob/dtheta_kdtheta_l = weight/(2T) sum_t sum_ij {[2 *
    % dw_ij/dtheta_k * dw_ij/dtheta_l * r'_j(t)^2] + [4 * w_ij *
    % dw_ij/dtheta_k * r'_j(t) * r''_j(t) * dx_j(t)/dtheta_l] + [4 * w_ij *
    % dw_ij/dtheta_l * r'_j(t) * r''_j(t) * dx_j(t)/dtheta_k] + [2 * w_ij^2
    % * r''_j(t)^2 * dx_j(t)/dtheta_k * dx_j(t)/dtheta_l] + [2 * w_ij^2 *
    % r'_j(t) * r'''_j(t) * dx_j(t)/dtheta_k * dx_j(t)/dtheta_l] + [2 *
    % w_ij^2 * r'_j(t) * r''_j(t) * d2x_j(t)/dtheta_kdtheta_l]}. In the {},
    % the first term can be calculated directly without backpropagation,
    % the second, third, fifth (note that the derivative may be negative)
    % and sixth terms are not positive semi-definite, and the fourth term
    % can be calculated through backpropagation. Only these two (i.e. the
    % first and the fourth) out of six terms are considered. The fourth
    % term is calculated here by noticing that the matrix-vector product
    % can be rewritten as J_x' * diag(2 * sum_i w_ij^2 * r''_j^2) * J_x *
    % v. So what we have to do is just backpropagate diag(2 * sum_i w_ij^2
    % * r''_j^2) * J_x * v = diag(2 * sum_i w_ij^2 * r''_j^2) * R{x}.
    
    frob_factor = frob_reg_weight/(2.0*(T+init_state_increase_t));
        
    n_d2r_t = zeros(N,T);
    n_d2r0_1 = zeros(N,1);
        
    % GPU-based computing can be implemented here to accelerate.
    n_d2r_t = rec_D2operator(n_r_t);
    n_d2r0_1 = rec_D2operator(n_r0_1);
        
    if isfield(net.frobenius_norm_regularizer, 'frob_row_idxs') % This allows to choose specific rows & columns.
        frob_row_idxs = net.frobenius_norm_regularizer.frob_row_idxs;
        frob_col_idxs = net.frobenius_norm_regularizer.frob_col_idxs;
    else
        frob_row_idxs = 1:N;
        frob_col_idxs = 1:N;
    end
    
    fr_Wrr_frob_fc = n_Wrr_n(frob_row_idxs, frob_col_idxs);
    fc_d2r_frob_tp1 = [n_d2r0_1(frob_col_idxs) n_d2r_t(frob_col_idxs,:)];
    fc_Rx_frob_tp1 = [n_Rx0_1(frob_col_idxs) n_Rx_t(frob_col_idxs,:)];
        
    sW2_fc = sum(fr_Wrr_frob_fc.^2,1);
    fc_RdLfrobdx_tp1 = bsxfun(@times, 2.0 * fc_d2r_frob_tp1.^2 .* fc_Rx_frob_tp1, (frob_factor * sW2_fc'));
    n_RdLfrobdx_tp1 = zeros(N,T+1);
    n_RdLfrobdx_tp1(frob_col_idxs,:) = fc_RdLfrobdx_tp1;
    n_RdLextra_tp1 = n_RdLextra_tp1 + n_RdLfrobdx_tp1;
end

if do_firing_rate_mean_regularizer
    % H_kl = d2L_fr_mean/dtheta_kdtheta_l = weight/(2N) * sum_i {2 * (1/T
    % sum_t dr_i(t)/dtheta_k) * (1/T sum_t' dr_i(t')/dtheta_l) + 2 * (1/T
    % sum_t r_i(t) - dv) * (1/T sum_t' d2r_i(t')/dtheta_kdtheta_l)}.
    % Consider only the positive semi-definite components, we have that H_L
    % can be approximately calculated as weight/(NT) * sum_t J_r(t),theta^T
    % * (1/T sum_t' J_r(t'),theta).
    rmean_factor = fr_mean_reg_weight / (num_fr_mean_reg * T);
    n_RdLfrmeandr_1 = rmean_factor * (mean(n_Rr_t,2) .* fr_mean_reg_mask);
    n_RdLextra_tp1(:,2:T+1) = n_RdLextra_tp1(:,2:T+1) + repmat(n_RdLfrmeandr_1, 1, T);
end

if do_firing_rate_var_regularizer
    rvar_factor = 2 * fr_var_reg_weight / (num_fr_var_reg * T);
    
    % H_kl = d2L_fr_var/dtheta_kdtheta_l = weight/(2N) * sum_i {2 * [1/T
    % sum_t 2 * rma_i(t) * dr_i(t)/dtheta_k] * [1/T sum_t' 2 * rma_i(t') *
    % dr_i(t')/dtheta_l] + 2 * [1/T sum_t' rma_i(t')^2 -dv] * [1/T sum_t 2
    % * drma_i(t)/dtheta_l * dr_i(t)/dtheta_k + 2 * rma_i(t) *
    % d2r_i(t)/dtheta_kdtheta_l]}. Consider only the positive semi-definite
    % components, we have that H_L can be approximately calculated as 2 *
    % weight/(NT) * {sum_t J_r(t),theta^T * [diag(rma(t)) * 2/T sum_t'
    % diag(rma(t')) * J_r(t'),theta] + sum_t J_r(t),theta^T * [1/T sum_t'
    % diag(rma(t'))^2 * (J_r(t'),theta - 1/T sum_t'' J_r(t''),theta)]}.
    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_var_reg_mask, 1, T);
    n_r_var_1 = (1/T)*sum(n_rma_t.^2, 2);

    n_Rr_avg_1 = (1/T)*sum(n_Rr_t, 2);
    n_Rrma_t = (n_Rr_t - repmat(n_Rr_avg_1, 1, T)) .* repmat(fr_var_reg_mask, 1, T);
    
    n_A_t = repmat((2/T) * sum((n_rma_t .* n_Rrma_t), 2), 1, T) .* n_rma_t;
    n_B_t = repmat(n_r_var_1, 1, T) .* n_Rrma_t;
    
    n_RdLfrvardr_t = rvar_factor * (n_A_t + n_B_t);
    n_RdLextra_tp1(:,2:T+1) = n_RdLextra_tp1(:,2:T+1) + n_RdLfrvardr_t;
end

if do_firing_rate_covar_regularizer
    rcovar_factor = 2 * fr_covar_reg_weight / (num_fr_covar_reg^2 * T);

    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
    n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
        
    n_Rr_avg_1 = (1/T)*sum(n_Rr_t, 2);
    n_Rrma_t = (n_Rr_t - repmat(n_Rr_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
    
    % H_kl = d2L_fr_covar/dtheta_kdtheta_l = weight/(2N^2) * 2 * sum_ij
    % {1/T sum_t [dr_i(t)/dtheta_k * rma_j(t) + dr_j(t)/dtheta_k *
    % rma_i(t)] * 1/T sum_t' [dr_i(t')/dtheta_l * rma_j(t') +
    % dr_j(t')/dtheta_l * rma_i(t')] + (cov_ij - dv * delta_ij) * 1/T sum_t
    % [d2r_i(t)/dtheta_kdtheta_l * rma_j(t) + dr_i(t)/dtheta_k *
    % (dr_j(t)/dtheta_l - 1/T sum_t' dr_j(t')/dtheta_l) +
    % d2r_j(t)/dtheta_kdtheta_l * rma_i(t) + dr_j(t)/dtheta_k *
    % (dr_i(t)/dtheta_l - 1/T sum_t' dr_i(t')/dtheta_l)]}. Consider only
    % the positive semi-definite components, we have that H_L can be
    % approximately calculated as 2 * weight/(N^2*T^2) * sum_t sum_t'
    % {J_r(t),theta^T * [(rma(t')^T * rma(t)) * I] * J_r(t'),theta +
    % J_r(t),theta^T * [rma(t') * rma(t)^T] * J_r(t'),theta} + 2 *
    % weight/(N^2*T) * sum_t J_r(t),theta^T * cov * J_r(t),theta.
    n_Rrma_x_rma_n = (1/T)*(n_Rrma_t * n_rma_t');
    n_rma_x_Rrma_n = (1/T)*(n_rma_t * n_Rrma_t');
    n_Rrcov_n = n_Rrma_x_rma_n + n_rma_x_Rrma_n;
    
    n_RdLfrcovardr_t = rcovar_factor * (n_Rrcov_n * n_rma_t + n_rcov_n * n_Rrma_t);
    n_RdLextra_tp1(:,2:T+1) = n_RdLextra_tp1(:,2:T+1) + n_RdLfrcovardr_t;
end
    
%% Do backward pass. Note that this is r1 pass (different from r2 which passes the R operator).
lambda_mu = lambda * mu;
n_RdLdx_t = zeros(N,T);
n_RdLdx_1 = zeros(N,1);

n_x_t = forward_pass{5};

for t = T:-1:1
    m_RdLdy_1 = m_RdLdy_t(:,t);
    n_RdLdx_1 = n_xdec_n * n_RdLdx_1 + n_dr_t(:,t) .* (n_Wrrt_dec_n * n_RdLdx_1 + n_Wrzt_m * m_RdLdy_1 + n_RdLextra_tp1(:,t+1));
        
    % Mind the structural damping. Note that we are IMPLICITLY assuming
    % that the distances between hidden states are measured by the loss
    % function which matches the transfer function of the hidden layer.
    % Also note that tanh is a rescaled version of the logistic function,
    % whose matching loss function is -sum_i {(r*_i+1)/2 log((r_i+1)/2) +
    % (1 - (r*_i+1)/2) log(1 - (r_i+1)/2)} and A = I, J_M = diag(1-r.^2).
    if mu > 0        
        n_RdLdx_1 = n_RdLdx_1 + (lambda_mu * n_Rr_t(:,t));
    end
    n_RdLdx_t(:,t) = n_RdLdx_1;
end

n_RdLdx0_1 = zeros(N,1);
if net.do_learn_init_state
    n_RdLdx0_1 = n_xdec_n * n_RdLdx_1 + n_dr0_1 .* (n_Wrrt_dec_n * n_RdLdx_1); % Just evaluate for one more step.
    if mu > 0
        n_RdLdx0_1 = n_RdLdx0_1 + (lambda_mu * n_Rr0_1);
    end
end

% Update the R{} w.r.t. the weights.
t_rt_n = n_r_t';
% m_RdLdy_t is vmasked above.
m_RdLdWzr_n = m_Wzr_entry_mask_n .* (m_RdLdy_t * t_rt_n);       
n_RdLdWru_v = n_Wru_entry_mask_v .* (dt_o_tau * n_RdLdx_t * v_u_t');
n_RdLdWrr_n = n_Wrr_entry_mask_n .* (dt_o_tau * n_RdLdx_t * n_rm1_t');

if net.do_learn_biases
    n_RdLdbx_1 = n_bx_entry_mask_1 .* (dt_o_tau * sum(n_RdLdx_t, 2));
    m_RdLdbz_1 = m_bz_entry_mask_1 .* sum(m_RdLdy_t, 2);
else
    n_RdLdbx_1 = zeros(N,1);
    m_RdLdbz_1 = zeros(M,1);
end

n_RdLdx0_c = zeros(N,net.nics);
if net.do_learn_init_state
    n_RdLdx0_c(:,condition) = n_x0_entry_mask_c(:,condition) .* n_RdLdx0_1;
end

% Here is the direct contribution of frobenius norm regularization.
if do_frobenius_norm_regularizer
    if isfield(net.frobenius_norm_regularizer, 'frob_row_idxs') % This allows to choose specific rows & columns.
        frob_row_idxs = net.frobenius_norm_regularizer.frob_row_idxs;
        frob_col_idxs = net.frobenius_norm_regularizer.frob_col_idxs;
    else
        frob_row_idxs = 1:N;
        frob_col_idxs = 1:N;
    end
    
    num_frob_reg = length(frob_row_idxs);
    fc_dr_frob_tp1 = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
    fr_VWrr_frob_fc = n_VWrr_n(frob_row_idxs,frob_col_idxs);
    
    frob_factor = frob_reg_weight/(2.0*(T+init_state_increase_t));
    sdr2_fc = sum(fc_dr_frob_tp1.^2,2)';
    fr_frobgv_fc = frob_factor * 2.0 * fr_VWrr_frob_fc .* repmat(sdr2_fc, num_frob_reg, 1);
    n_frobgv_n = zeros(N,N);
    n_frobgv_n(frob_row_idxs, frob_col_idxs) = fr_frobgv_fc;
    n_frobgv_n = n_Wrr_entry_mask_n .* n_frobgv_n;
    Gv = packRNN(net, n_RdLdWru_v, n_RdLdWrr_n + n_frobgv_n, m_RdLdWzr_n, n_RdLdx0_c, n_RdLdbx_1, m_RdLdbz_1);
else
    Gv = packRNN(net, n_RdLdWru_v, n_RdLdWrr_n, m_RdLdWzr_n, n_RdLdx0_c, n_RdLdbx_1, m_RdLdbz_1);
end

if do_norm_pres_regularizer
    npr_Wrr_n = n_Wrr_n(norm_pres_reg_mask,:);
    npr_VWrr_n = n_VWrr_n(norm_pres_reg_mask,:);
    npr_norm_pres_1 = vecnorm(n_Wrr_n(norm_pres_reg_mask,:)')';
    npr_norm2_pres_1 = npr_norm_pres_1.^2;
    
    norm_pres_factor = 2.0 * norm_pres_reg_weight / num_norm_pres_reg;
    
    n_gv_norm_pres_n = zeros(N,N);
    % d2L_norm_pres/dw_ijdw_kl = weight/(2N) d/dw_kl {sum_p 2 * [sum_q
    % w_pq^2 - dv^2] * [sum_q 2 * w_pq * dw_pq/dw_ij]} = 2 * weight/N sum_p
    % {2 * [sum_q w_pq * dw_pq/dw_ij] * [sum_q w_pq * dw_pq/dw_kl] + [sum_q
    % w_pq^2 - dv^2] * [sum_q dw_pq/dw_ij * dw_pq/dw_kl]} = 2 * weight/N *
    % [2 * delta_ik * w_ij * w_kl + delta_ik * delta_jl * (sum_q w_iq^2 -
    % dv^2)]. Consider only the elements on the diagonal line, we could
    % approximate gv as below.
    n_gv_norm_pres_n(norm_pres_reg_mask,:) = norm_pres_factor * ((2.0 * npr_Wrr_n.^2 .* npr_VWrr_n) + (repmat(npr_norm2_pres_1 - norm_pres_reg_dv^2, 1, N) .* npr_VWrr_n));
    % This is already masked because n_VWrr_n was masked above.
    % Pack it.
    gv_norm_pres = packRNN(net, zeros(N,V), n_gv_norm_pres_n, zeros(M,N), zeros(N,net.nics), zeros(N,1), zeros(M,1));
    Gv = Gv + gv_norm_pres;
end

if do_l2_regularizer
    Gv = Gv + l2_reg_weight * (modifiable_mask .* cost_mask .* v);
end

Gv = Gv + lambda * v; % Damping term (the classic Tikhonov regularization).

%% Numerically check the multiplication for debugging.
% This simple code depends on 1. matching loss functions and 2. all the
% parameters are being learned.
do_check_gv = 0;
epsilon = 1e-4; % Note that using overly small eps may cause the finite difference method to break down.
if do_check_gv && norm(Gv) > 0.001 && norm(v) > 0 && rand < 0.001
    disp(['Norm of Gv product: ' num2str(norm(Gv)) '.']);
    disp('Numerically checking Gv.');
    
    nparams = length(Gv);
    disp([num2str(nparams) ' params in total.']);

    theta = net.theta;
    testnetp = net;
    testnetm = net;
    
    p_G_p = zeros(nparams,nparams);
    m_Jzp_ts_p = cell(1,T);
    m_Jzpv_numer_t = zeros(M,T);
    n_Jxp_ts_p = cell(1,T);
    n_Jrp_ts_p = cell(1,T);
    
    xm_abs_diffs = zeros(1, nparams);
    xp_abs_diffs = zeros(1, nparams);
    
    % n_x_t = forward_pass{5};
    
    for i = 1:nparams
        e_i = zeros(nparams,1);
        e_i(i) = 1;
        
        theta_i_minus = theta - epsilon*e_i;
        theta_i_plus = theta + epsilon*e_i;
        
        testnetp.theta = theta_i_plus;
        testnetm.theta = theta_i_minus;
        
        package = eval_network(testnetp,condition,v_u_t,m_target_t);
        forward_pass_p = package{1};
        m_zp_nvm_t = forward_pass_p{3};
        m_zp_t = zeros(M,T);
        m_zp_t(vmask) = m_zp_nvm_t(vmask);
        n_rp_t = forward_pass_p{1};
        
        package = eval_network(testnetm,condition,v_u_t,m_target_t);
        forward_pass_m = package{1};
        m_zm_nvm_t = forward_pass_m{3};
        m_zm_t = zeros(M,T);
        m_zm_t(vmask) = m_zm_nvm_t(vmask);
        n_rm_t = forward_pass_m{1};
        
        n_xp_t = forward_pass_p{5};
        n_xm_t = forward_pass_m{5};
        
        xm_abs_diffs(i) = mean(vec(abs(n_xm_t - n_x_t)));
        xp_abs_diffs(i) = mean(vec(abs(n_xp_t - n_x_t)));
        
        for t = 1:T
            m_Jzp_ts_p{t}(:,i) = (m_zp_t(:,t)-m_zm_t(:,t))/(2.0*epsilon);
            n_Jxp_ts_p{t}(:,i) = (n_xp_t(:,t)-n_xm_t(:,t))/(2.0*epsilon);
            n_Jrp_ts_p{t}(:,i) = (n_rp_t(:,t)-n_rm_t(:,t))/(2.0*epsilon);
        end
        
        if mod(i,1000) == 0
            disp(['Calculated: ' num2str(i) ' params.']);
        end
    end
    
    for t = 1:T
        m_Jzpv_numer_t(:,t) = m_Jzp_ts_p{t} * v;
    end
    % Mask m_Rz_t first.
    m_Rz_vm_t = zeros(M,T);
    m_Rz_vm_t(vmask) = m_Rz_t(vmask);
    diff = m_Jzpv_numer_t - m_Rz_vm_t;
    disp(['Difference in J_(M.N)v = ' num2str(norm(diff)) '.']);

    switch net.objfun
        case 'sum-of-squares'
            m_hprime_1 = ones(M,1);
        case 'cross-entropy'
            switch net.layers(end).type
                case 'logistic'
                    assert(false, 'Function not implemented yet.');
                case 'softmax'
                    assert(false, 'Function not implemented yet.');
                otherwise
                    assert(false, ['Function ' net.layers(end).type ' not recognized.']);
            end
        otherwise
            assert(false, ['Function ' net.objfun ' not recognized.']);
    end
    switch net.layers(2).type % This is for structural damping.
        case 'linear'
            n_dprime_1 = ones(N,1);
        case 'tanh'
            n_dprime_1 = ones(N,1);
        otherwise
            assert(false, ['Function ' net.layers(2).type ' not implemented yet.']);
    end
    for t = 1:T
        % The normalization will be incorrect if not all the entries in a time slice are well-defined.
        p_G_p = p_G_p + (m_Jzp_ts_p{t}' * diag(m_hprime_1) * m_Jzp_ts_p{t}) / (ntargets/M);
        if mu > 0
            p_G_p = p_G_p + lambda_mu * n_Jxp_ts_p{t}' * diag(n_dprime_1) * n_Jrp_ts_p{t}; % J'_N * A * (J_M J_N v).
        end
    end
    
    numergv = p_G_p*v;
    if do_l2_regularizer
        netgv = Gv - l2_reg_weight * (modifiable_mask .* cost_mask .* v);
    end
    netgv = Gv - lambda * v;
    
    figure;
    stem(netgv,'r');
    hold on;
    stem(numergv,'b');
    
    figure;
    stem(xm_abs_diffs,'g'); % Just looking at magnitude here.
    hold on;
    stem(-xp_abs_diffs,'m');
    
    disp(['norm(numergv - netgv): ' num2str(norm(numergv-netgv)) '.']);
    disp(['norm(numergv - netgv)/norm(numergv + netgv): ' num2str(norm(numergv-netgv)/norm(numergv+netgv)) '.']);
    
    [n_Wru_ngv_v, n_Wrr_ngv_n, m_Wzr_ngv_n, n_x0_ngv_c, n_bx_ngv_1, m_bz_ngv_1] = unpackRNN(net, numergv);
    [n_Wru_gv_v, n_Wrr_gv_n, m_Wzr_gv_n, n_x0_gv_c, n_bx_gv_1, m_bz_gv_1] = unpackRNN(net, netgv);    
    disp(['Average differences in n_Wru_v: ' num2str(mean(vec(abs(n_Wru_ngv_v - n_Wru_gv_v))))]);
    disp(['Average differences in n_x0_c: ' num2str(mean(vec(abs(n_x0_ngv_c - n_x0_gv_c))))]);
    disp(['Average differences in n_Wrr_n: ' num2str(mean(vec(abs(n_Wrr_ngv_n - n_Wrr_gv_n))))]);
    disp(['Average differences in n_bx_1: ' num2str(mean(vec(abs(n_bx_ngv_1 - n_bx_gv_1))))]);
    disp(['Average differences in m_Wzr_n: ' num2str(mean(vec(abs(m_Wzr_ngv_n - m_Wzr_gv_n))))]);
    disp(['Average differences in m_bz_1: ' num2str(mean(vec(abs(m_bz_ngv_1 - m_bz_gv_1))))]);
    
    disp('The following should never be negative: ');
    disp(['vnumergv = ', num2str(dot(v, numergv))]);
    disp(['vnetgv = ', num2str(dot(v, netgv))]);
end

%% Return.
varargout = {};
varargout{end+1} = Gv;
varargout = {varargout};

end