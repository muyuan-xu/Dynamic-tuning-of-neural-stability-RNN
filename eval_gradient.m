function varargout = eval_gradient(net, init_condition, v_u_t, m_target_t, forward_pass)
%% Setup.
% Network structure.
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);
[V,T] = size(v_u_t);
[M,N] = size(m_Wzr_n);

% Transfer functions.
rec_transfun = net.layers(2).transfun;
out_transfun = net.layers(3).transfun;
rec_Doperator = net.layers(2).Doperator;
rec_D2operator = net.layers(2).D2operator;

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
    if (noise_sigma > 0.0)
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

%% Setup for backpropagation.
% Count the number of available target values.
vmask = ~isnan(m_target_t);
ntargets = length(find(vmask));

% Do backword pass to calculate gradient. 
% Take transpose to exchange pre and postsynaptic neurons.
if dt_o_tau < 1.0
    n_Wrrt_dec_n = dt_o_tau * n_Wrr_n';
    n_xdec_n = (1.0-dt_o_tau);
else
    n_Wrrt_dec_n = n_Wrr_n';
    n_xdec_n = 0;
end
n_Wrzt_m = m_Wzr_n';

m_dLdy_t = zeros(M,T);
if net.iscanonical
    % Rightnow only the sum-of-squares error function is implemented.
    m_dLdy_t(vmask) = m_z_t(vmask) - m_target_t(vmask);
else
    assert(false, 'Rightnow only the sum-of-squares error function is implemented.');
end
m_dLdy_t = m_dLdy_t / ntargets;

% Mind the regularizers.
n_dLextra_tp1 = zeros(N,T+1);
if do_frobenius_norm_regularizer
    % Note that there are two parts of the gradient: an indirect part
    % (dealt here, which needs backpropagation) and a direct part (dealt
    % below). L_frob = weight/(2T) sum_t sum_ij w_ij^2 * r'_j(t)^2. So use
    % the chain rule: dL_frob/dw_kl = weight/(2T) sum_t sum_ij {[2 * w_ij^2
    % * r'_j(t) * r''_j(t) * dx_j(t)/dw_kl] + [2 * w_ij * dw_ij/dw_kl *
    % r'_j(t)^2]}. The first component (in the {}) is calculated through
    % backpropagation, and the second component is calculated directly.
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
    fc_dr_frob_tp1 = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
    fc_d2r_frob_tp1 = [n_d2r0_1(frob_col_idxs) n_d2r_t(frob_col_idxs,:)];
    
    sW2_fc = sum(fr_Wrr_frob_fc.^2,1);
    fc_dLfrobdx_tp1 = frob_factor * bsxfun(@times, 2.0 * fc_dr_frob_tp1 .* fc_d2r_frob_tp1, sW2_fc');
    n_dLfrobdx_tp1 = zeros(N,T+1);
    n_dLfrobdx_tp1(frob_col_idxs,:) = fc_dLfrobdx_tp1;
    n_dLextra_tp1 = n_dLextra_tp1 + n_dLfrobdx_tp1;
end

if do_firing_rate_mean_regularizer
    % dL_fr_mean/dtheta_k = weight/(2N) * sum_i 2 * (1/T sum_t' r_i(t') -
    % dv) * (1/T sum_t dr_i(t)/dtheta_k). Rewrite in the vector form, we
    % have grad_L = weight/(NT) * sum_t J_r(t),theta^T * (1/T sum_t' r(t')
    % - dv).
    rmean_factor = fr_mean_reg_weight / (num_fr_mean_reg * T);
    n_dLfrmeandr_1 = rmean_factor * ((mean(n_r_t,2) - fr_mean_reg_dv) .* fr_mean_reg_mask);
    n_dLextra_tp1(:,2:T+1) = n_dLextra_tp1(:,2:T+1) + repmat(n_dLfrmeandr_1, 1, T);
end

if do_firing_rate_var_regularizer
    % dL_fr_var/dtheta_k = weight/(2N) * sum_i {2 * [1/T sum_t rma_i(t)^2 -
    % dv] * [1/T sum_t 2 * rma_i(t) * dr_i(t)/dtheta_k]}. Rewrite in the
    % vector form, we have grad_L = 2 * weight/(NT) * sum_t J_r(t), theta^T
    % * [(1/T sum_t rma(t).^2 - dv).*rma(t)].
    rvar_factor = 2 * fr_var_reg_weight / (num_fr_var_reg * T);
    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_rma_t = n_r_t - repmat(n_r_avg_1, 1, T);
    n_r_var_1 = (1/T)*sum(n_rma_t.^2, 2);
    n_dLfrvardr_t = rvar_factor * (repmat((n_r_var_1 - fr_var_reg_dv) .* fr_var_reg_mask, 1, T) .* n_rma_t);
    n_dLextra_tp1(:,2:T+1) = n_dLextra_tp1(:,2:T+1) + n_dLfrvardr_t;
end

if do_firing_rate_covar_regularizer
    % dL_fr_covar/dtheta_k = weight/(2N^2) * 2 * sum_ij (cov_ij - dv *
    % delta_ij) * 1/T sum_t [dr_i(t)/dtheta_k * rma_j(t) + dr_j(t)/dtheta_k
    % * rma_i(t)]. Rewrite in the vector form, we have grad_L = 2 *
    % weight/(N^2*T) * sum_t J_r(t),theta^T * (cov - dv * I) * rma(t).
    rcovar_factor = 2 * fr_covar_reg_weight / (num_fr_covar_reg^2 * T);
    n_dvI_n = (fr_covar_reg_dv * eye(N)) .* repmat(fr_covar_reg_mask, 1, N);
    n_r_avg_1 = (1/T)*sum(n_r_t, 2);
    n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
    n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
    n_dLfrcovardr_t = rcovar_factor * ((n_rcov_n - n_dvI_n) * n_rma_t);
    n_dLextra_tp1(:,2:T+1) = n_dLextra_tp1(:,2:T+1) + n_dLfrcovardr_t;
end

%% Do backward pass.
n_dLdx_t = zeros(N,T);
n_dLdx_1 = zeros(N,1);
for t = T:-1:1
    m_dLdy_1 = m_dLdy_t(:,t);
    % Rightnow the transfer function for the output layer is assumed to be linear.
    n_dLdx_1 = n_xdec_n * n_dLdx_1 + n_dr_t(:,t) .* (n_Wrrt_dec_n * n_dLdx_1 + n_Wrzt_m * m_dLdy_1 + n_dLextra_tp1(:,t+1));
    n_dLdx_t(:,t) = n_dLdx_1;
end
% Gradient w.r.t. the initial state of the RNN.
n_dLdx0_1 = zeros(N,1);
if net.do_learn_init_state
    n_dLdx0_1 = n_xdec_n * n_dLdx_1 + n_dr0_1 .* (n_Wrrt_dec_n * n_dLdx_1);
end

% Update the gradient w.r.t. weights.
[n_Wru_entry_mask_v, n_Wrr_entry_mask_n, m_Wzr_entry_mask_n, n_x0_entry_mask_c, n_bx_entry_mask_1, m_bz_entry_mask_1] = unpackRNNUtils(net, 'do_entry_mask', true);
t_rt_n = n_r_t';
% m_dLdy_t is already vmasked.
n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];
m_dLdWzr_n = m_Wzr_entry_mask_n .* (m_dLdy_t * t_rt_n);
n_dLdWru_v = n_Wru_entry_mask_v .* (dt_o_tau * n_dLdx_t * v_u_t');
n_dLdWrr_n = n_Wrr_entry_mask_n .* (dt_o_tau * n_dLdx_t * n_rm1_t');

n_dLdx0_c = zeros(N,net.nics);
n_dLdx0_c(:,init_condition) = n_x0_entry_mask_c(:,init_condition) .* n_dLdx0_1;

if net.do_learn_biases
    n_dLdbx_1 = n_bx_entry_mask_1 .* (dt_o_tau * sum(n_dLdx_t, 2));
    m_dLdbz_1 = m_bz_entry_mask_1 .* sum(m_dLdy_t, 2);
else
    n_dLdbx_1 = zeros(N,1);
    m_dLdbz_1 = zeros(M,1);
end

grad = packRNN(net, n_dLdWru_v, n_dLdWrr_n, m_dLdWzr_n, n_dLdx0_c, n_dLdbx_1, m_dLdbz_1);

if do_l2_regularizer
    grad = grad + l2_reg_weight * (modifiable_mask .* cost_mask .* net.theta);
end

% Non-backpropagate part of the frobenius norm regularizer.
if do_frobenius_norm_regularizer
    if isfield(net.frobenius_norm_regularizer, 'frob_row_idxs') % This allows to choose specific rows & columns.
        frob_row_idxs = net.frobenius_norm_regularizer.frob_row_idxs;
        frob_col_idxs = net.frobenius_norm_regularizer.frob_col_idxs;
    else
        frob_row_idxs = 1:N;
        frob_col_idxs = 1:N;
    end
    
    num_frob_reg = length(frob_row_idxs);
    fr_Wrr_frob_fc = n_Wrr_n(frob_row_idxs, frob_col_idxs);
    fc_dr_frob_tp1 = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
    fc_dr2_frob_tp1 = fc_dr_frob_tp1.^2;

    frob_factor = frob_reg_weight/(2.0*(T+init_state_increase_t));
    sdr2_fc = sum(fc_dr2_frob_tp1, 2)';
    fr_grad_frob_fc = frob_factor * (2.0 * fr_Wrr_frob_fc .* repmat(sdr2_fc, num_frob_reg, 1));
    n_grad_frob_n = zeros(N,N);
    n_grad_frob_n(frob_row_idxs,frob_col_idxs) = fr_grad_frob_fc;
    n_grad_frob_n = n_Wrr_entry_mask_n .* n_grad_frob_n;

    % Pack it.
    grad_frob = packRNN(net, zeros(N,V), n_grad_frob_n, zeros(M,N), zeros(N,net.nics), zeros(N,1), zeros(M,1));
    grad = grad + grad_frob;
end

if do_norm_pres_regularizer
    % dL_norm_pres/dw_kl = weight/(2N) * sum_i {2 * [sum_j w_ij^2 - dv^2] *
    % [sum_j 2 * w_ij * dw_ij/dw_kl]}. Rewrite in the vector form, we have
    % grad_L = 2 * weight/N * (diag_i(sum_j w_ij^2 - dv^2) * w).
    npr_Wrr_n = n_Wrr_n(norm_pres_reg_mask,:);
    npr_norm_pres_1 = vecnorm(n_Wrr_n(norm_pres_reg_mask,:)')';
    npr_norm2_pres_1 = npr_norm_pres_1.^2;
    norm_pres_factor = 2.0 * norm_pres_reg_weight / num_norm_pres_reg;
    
    n_grad_norm_pres_n = zeros(N,N); 
    n_grad_norm_pres_n(norm_pres_reg_mask,:) = norm_pres_factor * (repmat(npr_norm2_pres_1 - norm_pres_reg_dv^2, 1, N) .* npr_Wrr_n);
    n_grad_norm_pres_n = n_Wrr_entry_mask_n .* n_grad_norm_pres_n;

    % Pack it.
    grad_norm_pres = packRNN(net, zeros(N,V), n_grad_norm_pres_n, zeros(M,N), zeros(N,net.nics), zeros(N,1), zeros(M,1));
    grad = grad + grad_norm_pres;
end

%% Numerically check the gradient for debugging.
do_check_grad = 0;
epsilon = 1e-4;
if do_check_grad && norm(grad) > 0.01 && rand < 0.1
    disp(['Norm of the gradient: ' num2str(norm(grad)) '.']);
    disp('Numerically checking the gradient.');

    theta = net.theta;
    numergrad = zeros(size(theta));
    ngrads = size(theta(:),1);
    disp([num2str(ngrads) ' params in total.']);
    objfun = @(net) eval_objfun(net, init_condition, v_u_t, m_target_t, []);

    for i = 1:ngrads
        e_i = zeros(ngrads,1);
        e_i(i) = 1;
        theta_i_plus = theta + epsilon*e_i;
        theta_i_minus = theta - epsilon*e_i;
        
        testnetp = net;
        testnetm = net;
        testnetp.theta = theta_i_plus;
        testnetm.theta = theta_i_minus;
        package = objfun(testnetp);
        gradp = package{1};
        package = objfun(testnetm);
        gradm = package{1};
        numergrad(i) = (gradp - gradm)/(2.0*epsilon);
        
        if mod(i,1000) == 0
            disp(['Calculated: ' num2str(i) ' params.']);
        end
    end

    figure;
    stem(grad,'r')
    hold on
    stem(numergrad,'b');
    [n_Wru_ng_v, n_Wrr_ng_n, m_Wzr_ng_n, n_x0_ng_c, n_bx_ng_1, m_bz_ng_1] = unpackRNN(net, numergrad);
    [n_Wru_g_v, n_Wrr_g_n, m_Wzr_g_n, n_x0_g_c, n_bx_g_1, m_bz_g_1] = unpackRNN(net, grad);
    % I believe these results are HIGHLY dependent on whether or not the
    % gradient is exploding or vanishing. If the gradient is exploding,
    % this can look very, very ugly. Makes one wonder how this works at
    % all! If the gradient is vanishing, then this looks very, very good:
    % order 1e-10. Otherwise, it's down to 1e-5 or worse.
    disp(['Average differences in n_Wru_v: ' num2str(mean(vec(abs(n_Wru_ng_v - n_Wru_g_v))))]);
    disp(['Average differences in n_x0_c: ' num2str(mean(vec(abs(n_x0_ng_c - n_x0_g_c))))]);
    disp(['Average differences in n_Wrr_n: ' num2str(mean(vec(abs(n_Wrr_ng_n - n_Wrr_g_n))))]);
    disp(['Average differences in n_bx_1: ' num2str(mean(vec(abs(n_bx_ng_1 - n_bx_g_1))))]);
    disp(['Average differences in m_Wzr_n: ' num2str(mean(vec(abs(m_Wzr_ng_n - m_Wzr_g_n))))]);
    disp(['Average differences in m_bz_1: ' num2str(mean(vec(abs(m_bz_ng_1 - m_bz_g_1))))]);
    diff = norm(numergrad-grad)/norm(numergrad+grad);
    disp('Norm of the difference between numerical and analytical gradient should be < 1e-9:');
    disp(['norm(numgrad-grad)/norm(numgrad+grad): ' num2str(diff)]);
end

%% Return.
varargout = {};
varargout{end+1} = grad;
varargout = {varargout};

end