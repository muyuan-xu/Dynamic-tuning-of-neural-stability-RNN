function varargout = eval_network(net, condition, v_u_t, m_target_t)
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

%% Do forward pass.
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

%% Return.
varargout = {};
varargout{end+1} = {n_r_t n_dr_t m_z_t n_r0_1 n_x_t n_x0_1};
varargout = {varargout};

end