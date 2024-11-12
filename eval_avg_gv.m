function varargout = eval_avg_gv(net, conditions_s, inputs_s, targets_s, v, lambda, forward_passes_s, varargin)
%% Calculate the averaged matrix-vector product.
do_parallel = false;
optargin = size(varargin,2);
for i = 1:2:optargin
    switch varargin{i}
        case 'do_parallel'
            do_parallel = varargin{i+1};
        otherwise
            assert(false,['Variable argument ' varargin{i} ' not recognized.']);
    end
end

if ~isempty(inputs_s)
    ntrials = size(inputs_s,2);
elseif ~isempty(targets_s)
    ntrials = size(targets_s,2);
end

avg_gv = zeros(size(net.theta));
if ~do_parallel
    for i = 1:ntrials
        package = eval_gv(net,conditions_s{i},inputs_s{i},targets_s{i},v,lambda,forward_passes_s{i});
        avg_gv = avg_gv + package{1};
    end
else
    parfor i = 1:ntrials
        package = eval_gv(net,conditions_s{i},inputs_s{i},targets_s{i},v,lambda,forward_passes_s{i});
        avg_gv = avg_gv + package{1};
    end
end
avg_gv = avg_gv / ntrials;

%% Return.
varargout = {};
varargout{end+1} = avg_gv;
varargout = {varargout};

end