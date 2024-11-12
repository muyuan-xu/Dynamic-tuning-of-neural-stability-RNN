function [q, n_gradq_1, n_Gq_n] = find_one_fp(net, n_x_1, const_input, do_topo_map, tolq)
% Return the function q(x) = 1/2 |F(x)|^2 at n_x_1, where dx/dt = F(x)
% describes the (continuous) dynamics of RNN (note that time constant tau
% is absorbed in F). Also return its gradient and the Gauss Newton
% approximation to its Hessian.
%
% const_input allows the fixed point to depend on input and tolq allows us
% to manually stop the optimization (after q falling below tolq) in case of
% searching for iso-speed contours (do_topo_map = true).

N = net.layers(2).npost;
[n_Wru_v, n_Wrr_n, ~, ~, n_bx_1, ~] = unpackRNN(net, net.theta);

dt_o_tau = net.dt / net.tau;

n_r_1 = net.layers(2).transfun(n_x_1);

if dt_o_tau < 1.0
    % dx/dt = F(x) = (-x + J*r + B*u + b) / tau.
    n_Fx_1 = -n_x_1 + n_Wrr_n * n_r_1 + n_bx_1;
    if(~isempty(const_input))
        n_Fx_1 = n_Fx_1 + (n_Wru_v * const_input);
    end
    n_Fx_1 = n_Fx_1 / net.tau;
else
    % x(t+1) = G(x) = J*r(t) + B*u(t+1) + b.
    n_Gx_1 = n_Wrr_n * n_r_1 + n_bx_1;
    if(~isempty(const_input))
        n_Gx_1 = n_Gx_1 + (n_Wru_v * const_input);
    end
    n_Fx_1 = n_Gx_1 - n_x_1;
end

q = 0.5 * (n_Fx_1'*n_Fx_1);

n_gradq_1 = zeros(N,1);
n_Gq_n = zeros(N,N);

if ~do_topo_map || (do_topo_map && q >= tolq)
    n_dr_1 = net.layers(2).Doperator(n_r_1);
    if dt_o_tau < 1.0
        % Jacobian.
        n_J_F_n = (-eye(N) + n_Wrr_n * diag(n_dr_1)) / net.tau;
    else
        % Note that this is the Jacobian of G(x(t)) - G(x(t-1)).
        n_J_F_n = -eye(N) + n_Wrr_n * diag(n_dr_1);
    end
    n_gradq_1 = n_J_F_n' * n_Fx_1;
    n_Gq_n = n_J_F_n' * n_J_F_n;
end

end