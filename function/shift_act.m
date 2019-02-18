function activation_shift = shift_act(activation,pattern)
    numDimsA = ndims(activation);
    pattern(numDimsA) = 0;
    idx = cell(1, numDimsA);

    for k = 1:numDimsA-2
        m = size(activation,k);
        idx_t = (1:m)-pattern(k);
        idx_t(idx_t<1) = 1;
        idx_t(idx_t>m) = m;
        idx{k} = idx_t;
    end
    idx{3} = ':';
    idx{4} = ':';

    activation_shift = activation(idx{:});
end
