function theta = packRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1)
% This function is appropriate for an FFN with a single hidden layer. The
% variable names are w.r.t. to the actual weights, but any derivative
% products during learning can be packed / unpacked in this way. Note that
% net isn't modified or returned. This accomodates the R{} technique
% implementation. Also note that the first layer is a fake input and its
% bias is used as the x(0) for multiple conditions.

nparams = sum([net.layers.nparams]);
nparams_cumsum = cumsum([net.layers.nparams]);
nparams_cumsum = [0 nparams_cumsum];

W{1} = n_Wru_v;
bias{1} = n_x0_c;
W{2} = n_Wrr_n;
bias{2} = n_bx_1;
W{3} = m_Wzr_n;
bias{3} = m_bz_1;

theta = zeros(nparams,1);
for i = 1:3
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);
    theta(layer_start_idx:layer_stop_idx) = [vec(W{i});vec(bias{i})]; % Align params in a vector.
end

end