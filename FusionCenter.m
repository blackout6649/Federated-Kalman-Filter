classdef FusionCenter
    % Information-form track-to-track fusion assuming independence
    methods (Static)
        function [x_fused, P_fused] = fuse(estimates, covariances)
            % estimates: cell{N} of x_i
            % covariances: cell{N} of P_i
            % weights: optional vector (default: all ones)
            N = numel(estimates); % Number of sensors
            n = numel(estimates{1});  % Number of states in for each sensor
            Y = zeros(n);             % Initialize fused information matrix
            y = zeros(n,1);           % Initialize fused information vector
            for i = 1:N
                Pi = covariances{i};
                xi = estimates{i};
                Yi = Pi \ eye(n);       % robust inverse
                Y = Y + Yi;
                y = y + Yi * xi;
            end
            P_fused = Y \ eye(n);
            x_fused = P_fused * y;
        end
    end
end
