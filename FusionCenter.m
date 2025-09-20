classdef FusionCenter
    % Information-form track-to-track fusion assuming independence
    methods (Static)
        function [x_fused, P_fused] = information_fusion(X, P, weights)
        % Implementation of fusion with custom weights
        % X: cell array of states
        % P: cell array of covariances
        % weights: array of fusion weights, must sum to 1
        
        % Federated filter fusion equation
        % P_fused_inv = sum(P_i_inv * beta_i)
        % x_fused = P_fused * sum(P_i_inv * x_i * beta_i)
        
        I_fused = zeros(size(P{1}));
        y_fused = zeros(size(X{1}));
        
        for i = 1:numel(X)
            P_inv = inv(P{i});
            I_fused = I_fused + weights(i) * P_inv;
            y_fused = y_fused + weights(i) * P_inv * X{i};
        end
        
        P_fused = inv(I_fused);
        x_fused = P_fused * y_fused;
        end

        function [x_fused, P_fused] = fuse(estimates, covariances)
            % estimates: cell{N} of x_i
            % covariances: cell{N} of P_i
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
