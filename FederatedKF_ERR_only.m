classdef FederatedKF_ERR_only < handle
    % Orchestrates multiple LocalKalmanFilter objects + fusion center with fault detection
    properties
        locals   % array of LocalKalmanFilter
        weight   % fusion weights (optional)
        x        % fused state
        P        % fused covariance
        name
        
        % Fault detection properties
        chi_square_threshold   % chi-square test thresholds for each local filter
        fault_flags            % current fault status for each local filter (0=fault, 1=normal)
        confidence_level       % confidence level for chi-square test (default 0.05)
        method                 % Method of handling errors ( 1 = Assign last global (fused) value | 2 = Skip update | 3 = Assign last global (fused) value and increase cov)
        
        % Sliding window properties for enhanced detection
        window_size          % sliding window size (default 10)
        residual_history     % cell array storing residual history for each local
        covar_history        % cell array storing residual covariance history
    end
    
    methods
        function obj = FederatedKF_ERR_only(locals, weight, name, confidence_level, window_size, method) % Initialization
            if nargin < 2, weight = []; end
            if nargin < 3, name = "FKF"; end
            if nargin < 4, confidence_level = 0.05; end  % 95% confidence
            if nargin < 5, window_size = 10; end
            if nargin < 6, method = 1; end
            
            obj.locals = locals; 
            obj.weight = weight; 
            obj.name = name;
            obj.confidence_level = confidence_level;
            obj.window_size = window_size;
            obj.method = method;
            
            % Initialize fault detection arrays
            num_locals = numel(obj.locals);
            obj.fault_flags = ones(1, num_locals);  % all start as normal
            obj.chi_square_threshold = zeros(1, num_locals);
            obj.residual_history = cell(1, num_locals);
            obj.covar_history = cell(1, num_locals);
            
            % Initialize thresholds based on measurement dimensions
            for i = 1:num_locals
                m = obj.locals(i).sensor.measDimension;
                obj.chi_square_threshold(i) = chi2inv(1 - confidence_level, m);
                obj.residual_history{i} = [];
                obj.covar_history{i} = [];
            end
            
            % Initialize fused with first local
            [x0, P0] = obj.locals(1).estimate();
            obj.x = x0; 
            obj.P = P0;
        end
        
        function step(obj, z_cell, fuseFlag)
            % z_cell: cell{N} of measurements for each local (or [] to skip)
            
            % 1) local predicts
            for i = 1:numel(obj.locals)
                obj.locals(i).predict();
            end
            
            % 2) fault detection and local updates
            for i = 1:numel(obj.locals)
                zi = z_cell{i};
                if all(~isnan(zi))
                    % Perform fault detection before update
                    obj.detectFault(i, zi);
                    
                    % Only update if no fault detected
                    if obj.fault_flags(i) == 1
                        obj.locals(i).update(zi);
                    else
                        switch obj.method
                            case 1
                                fprintf('Fault detected in local filter %d - assigning last global value\n', i);
                                obj.locals(i).x = obj.x;
                            case 2 
                                fprintf('Fault detected in local filter %d - skipping update\n', i);
                            case 3 
                                fprintf('Fault detected in local filter %d - assigning last global value and increasing covariance\n', i);
                                obj.locals(i).x = obj.x;
                                obj.locals(i).P = obj.P * numel(obj.locals);
                        end 
                    end
                end
            end
            
            % 3) fusion (skip if haven't reached fusion step)
            if fuseFlag
                valid_indices = find(obj.fault_flags == 1);
                
                if isempty(valid_indices)
                    warning('All local filters have faults - skipping fusion');
                    return;
                end
                
                % Pre-allocate cells for only the valid filters
                num_valid = numel(valid_indices);
                X = cell(1, num_valid);
                P = cell(1, num_valid);
                
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    [X{j}, P{j}] = obj.locals(local_idx).estimate();
                end
                
                % Fuse only the valid estimates
                [obj.x, obj.P] = FusionCenter.fuse(X, P);
                
                % Reset local filters
                for i = 1:numel(obj.locals)
                    obj.locals(i).x = obj.x;
                    obj.locals(i).P = obj.P * obj.weight;
                end
            end 
        end
        
        function detectFault(obj, local_idx, measurement)
            % chi-square fault detection 
            
            local_filter = obj.locals(local_idx);
            
            % Get measurement model parameters
            H = local_filter.sensor.H;  % measurement matrix
            R = local_filter.sensor.R;  % measurement noise covariance
            x_pred = local_filter.x;    % predicted state
            P_pred = local_filter.P;    % predicted covariance
            
            % Predicted measurement
            z_pred = H * x_pred;
            
            % Residual
            residual = measurement - z_pred;
            
            % Residual covariance 
            residual_covar = H * P_pred * H' + R;
            
            % Chi-square test statistic 
            lambda = residual' / residual_covar * residual;
            
            % Basic chi-square test 
            if lambda >= obj.chi_square_threshold(local_idx)
                obj.fault_flags(local_idx) = 0;  % fault detected
            else
                obj.fault_flags(local_idx) = 1;  % normal operation
            end
            
            % Enhanced detection with sliding window averaging 
            obj.updateSlidingWindow(local_idx, residual, residual_covar);
            obj.slidingWindowTest(local_idx);
        end
        
        function updateSlidingWindow(obj, local_idx, residual, theoretical_covar)
            % Update residual history for sliding window analysis
            
            residual_outer = residual * residual';
            
            % Add to history
            obj.residual_history{local_idx} = [obj.residual_history{local_idx}, {residual_outer}];
            obj.covar_history{local_idx} = [obj.covar_history{local_idx}, {theoretical_covar}];
            
            % Maintain window size
            if length(obj.residual_history{local_idx}) > obj.window_size
                obj.residual_history{local_idx} = obj.residual_history{local_idx}(2:end);
                obj.covar_history{local_idx} = obj.covar_history{local_idx}(2:end);
            end
        end
        
        function slidingWindowTest(obj, local_idx)
            % Implements sliding window test 
            
            history = obj.residual_history{local_idx};
            covar_history = obj.covar_history{local_idx};
            
            if length(history) < obj.window_size
                return;  % Not enough data yet
            end
            
            % Calculate actual residual covariance
            actual_covar = zeros(size(history{1}));
            theoretical_covar = zeros(size(covar_history{1}));
            
            for i = 1:length(history)
                actual_covar = actual_covar + history{i};
                theoretical_covar = theoretical_covar + covar_history{i};
            end
            
            actual_covar = actual_covar / length(history);
            theoretical_covar = theoretical_covar / length(history);
            
            % Calculate deviation ratio 
            eta = trace(theoretical_covar) / trace(actual_covar);
            
            % Enhanced fault detection criteria
            if eta > 2 || eta < 0.2
                obj.fault_flags(local_idx) = 0;  % fault detected
            elseif eta >= 0.5 && eta <= 1.5
                % Within normal range - could override basic chi-square if needed
                obj.fault_flags(local_idx) = 1;
            end
        end
        
        function status = getFaultStatus(obj)
            % Returns current fault status of all local filters
            status = obj.fault_flags;
        end
        
        function printFaultStatus(obj)
            % Prints current fault status
            fprintf('Fault Status: ');
            for i = 1:length(obj.fault_flags)
                if obj.fault_flags(i) == 1
                    fprintf('L%d:OK ', i);
                else
                    fprintf('L%d:FAULT ', i);
                end
            end
            fprintf('\n');
        end
    end
end