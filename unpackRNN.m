function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, theta)
% Note that net.theta is NOT used. If you want to use this function with
% net.theta, then pass it as the second parameter. This accomodates the R{}
% technique for exact computation of the Hessian.

nparams = sum([net.layers.nparams]);
nparams_cumsum = cumsum([net.layers.nparams]);
nparams_cumsum = [0 nparams_cumsum];

nics = net.nics;
for i = 1:3
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);

    npost = net.layers(i).npost;
    npre = net.layers(i).npre;
    
    W_and_b  = theta(layer_start_idx:layer_stop_idx);
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