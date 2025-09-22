classdef FederatedKF < handle
    % Orchestrates multiple LocalKalmanFilter objects + fusion center.
    % Implements a two-phase approach: a 'calibration' phase to learn
    % sensor weights using a reference filter, followed by an 'operational' phase.
    
    properties
        locals   % array of LocalKalmanFilter
        ISF      % Information Sharing Factors (learned during calibration)
        x        % fused state
        P        % fused covariance
        name
        
        % Fault detection properties
        chi_square_threshold
        fault_flags
        confidence_level
        method
        
        % Sliding window properties
        window_size
        residual_history
        covar_history
        
        % --- Calibration Properties ---
        mode                 % 'calibration' or 'operational'
        reference_filter     % The trusted "System B" filter
        calibration_duration % Number of steps for the calibration phase
        calibration_steps    % Counter for the current calibration step
        calibration_errors   % Stores accumulated squared errors for each local filter
        
        % --- Separate MSE tracking for operational phase ---
        operational_errors   % Tracks errors during operational phase for ISF updates
        operational_steps    % Counter for operational phase steps

        % --- Adjusted reference weight property ---
        reference_weight     % Manual weight of the reference filter (reduced default)
    end
    
    methods
        % --- Constructor with better defaults ---
        function obj = FederatedKF(locals, reference_filter, calibration_duration, name, confidence_level, window_size, method, reference_weight)
            if nargin < 4, name = "CalibratingFKF"; end
            if nargin < 5, confidence_level = 0.05; end
            if nargin < 6, window_size = 10; end
            if nargin < 7, method = 1; end
            if nargin < 8, reference_weight = 0.3; end
            
            obj.locals = locals;
            obj.name = name;
            obj.confidence_level = confidence_level;
            obj.window_size = window_size;
            obj.method = method;
            obj.reference_weight = reference_weight;
            
            num_locals = numel(obj.locals);
            
            % --- Initialize Calibration ---
            obj.mode = 'calibration';
            obj.reference_filter = reference_filter;
            obj.calibration_duration = calibration_duration;
            obj.calibration_steps = 0;
            obj.calibration_errors = zeros(1, num_locals);
            
            % --- Initialize operational tracking ---
            obj.operational_errors = zeros(1, num_locals);
            obj.operational_steps = 0;
            
            % --- Initialize ISF ---
            obj.ISF = zeros(1, num_locals);
 
            
            fprintf('--- FKF "%s" initialized in CALIBRATION mode for %d steps. ---\n', obj.name, obj.calibration_duration);
            
            % Initialize fault detection arrays
            obj.fault_flags = ones(1, num_locals);
            obj.chi_square_threshold = zeros(1, num_locals);
            obj.residual_history = cell(1, num_locals);
            obj.covar_history = cell(1, num_locals);
            
            for i = 1:num_locals
                m = obj.locals(i).sensor.measDimension;
                obj.chi_square_threshold(i) = chi2inv(1 - confidence_level, m);
            end
            
            [x0, P0] = obj.locals(1).estimate();
            obj.x = x0;
            obj.P = P0;
        end
        
        function step(obj, z_cell, z_ref, fuseFlag)
            % z_cell: cell{N} of measurements for local filters
            % z_ref: measurement for the reference filter
            
            if strcmp(obj.mode, 'calibration')
                obj.calibrationStep(z_cell, z_ref);
                if fuseFlag
                    obj.fuseFilters();
                end
            else % operational mode
                obj.operationalStep(z_cell, z_ref, fuseFlag);
            end
        end
        
        function calibrationStep(obj, z_cell, z_ref)
            % Predict all local filters and the reference filter
            obj.reference_filter.predict();
            for i = 1:numel(obj.locals)
                obj.locals(i).predict();
            end
            
            % Update all filters
            if all(~isnan(z_ref))
                obj.reference_filter.update(z_ref);
            end
            for i = 1:numel(obj.locals)
                if all(~isnan(z_cell{i}))
                    obj.locals(i).update(z_cell{i});
                end
            end
            
            % Accumulate error against the reference
            ref_state = obj.reference_filter.estimate();
            for i = 1:numel(obj.locals)
                [local_state, ~] = obj.locals(i).estimate();
                error_vec = local_state - ref_state;
                obj.calibration_errors(i) = obj.calibration_errors(i) + (error_vec' * error_vec);
            end
            
            obj.calibration_steps = obj.calibration_steps + 1;
            fprintf('Calibration step %d/%d...\n', obj.calibration_steps, obj.calibration_duration);
            
            % Check if calibration is complete
            if obj.calibration_steps >= obj.calibration_duration
                obj.finalizeCalibration();
            end
        end
        
        % --- Filter fusion ---
        function fuseFilters(obj)
            fprintf('Fusing filters...\n');
            
            valid_indices = find(obj.fault_flags == 1);
            if isempty(valid_indices)
                warning('All local filters have faults - cannot perform fusion');
                return;
            end
            
            num_valid = numel(valid_indices);
            
            if strcmp(obj.mode, 'calibration')
                % --- CALIBRATION MODE: Include reference filter ---
                X = cell(1, num_valid + 1); 
                P = cell(1, num_valid + 1);
                ISF_valid = zeros(1, num_valid + 1);
                
                % Add reference filter to the fusion set
                [X{1}, P{1}] = obj.reference_filter.estimate();
                ISF_valid(1) = obj.reference_weight; 
                
                % Add local filters to the fusion set
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    [X{j+1}, P{j+1}] = obj.locals(local_idx).estimate();
                    ISF_valid(j+1) = obj.ISF(local_idx);
                end
                
                % Normalize the local weights so they sum to (1 - reference_weight)
                sum_of_other_ISF = sum(ISF_valid(2:end));
                if sum_of_other_ISF > 1e-6
                    ISF_valid(2:end) = ISF_valid(2:end) * (1 - obj.reference_weight) / sum_of_other_ISF;
                else
                    ISF_valid(2:end) = (1 - obj.reference_weight) / num_valid;
                end
                
                % Perform fusion using the weights
                [obj.x, obj.P] = FusionCenter.information_fusion(X, P, ISF_valid);
 
            else
                % --- OPERATIONAL MODE: Local filters only with normalized weights ---
                X = cell(1, num_valid);
                P = cell(1, num_valid);
                
                % Get valid local filters and their ISFs
                total_ISF = 0;
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    [X{j}, P{j}] = obj.locals(local_idx).estimate();
                    total_ISF = total_ISF + obj.ISF(local_idx);
                end
                
                % Normalize weights to sum to 1
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    normalized_weights(j) = obj.ISF(local_idx) / total_ISF;
                end
                
                % Perform fusion with normalized weights
                [obj.x, obj.P] = FusionCenter.information_fusion(X, P, normalized_weights);
                      
                fprintf('Operational fusion with  weights: ');
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    fprintf('L%d:%.3f ', local_idx, obj.ISF(j));
                end
                fprintf('\n');

            end
            % Share information with local filters
            for j = 1:num_valid 
                local_idx = valid_indices(j);
                obj.locals(local_idx).P = obj.P;
                obj.locals(local_idx).x = obj.x;
            end 
        end
        
        % % --- Information sharing reset ---
        % function shareInformation(obj, valid_indices)
        %     % Share global information with local filters using information form
        %     global_info = inv(obj.P);
        %     global_info_vec = global_info * obj.x;
        % 
        %     for j = 1:numel(valid_indices)
        %         i = valid_indices(j);
        % 
        %         % Get current local estimate
        %         [x_local, P_local] = obj.locals(i).estimate();
        %         local_info = inv(P_local);
        %         local_info_vec = local_info * x_local;
        % 
        %         % Calculate information to share based on ISF
        %         % alpha = obj.ISF(i) / numel(obj.locals);  % Normalize ISF
        %         alpha = obj.ISF(i);
        %         info_to_share = alpha * (global_info - local_info);
        %         info_vec_to_share = alpha * (global_info_vec - local_info_vec);
        % 
        %         % Update local filter with shared information
        %         new_info = local_info + info_to_share;
        %         new_info_vec = local_info_vec + info_vec_to_share;
        % 
        %         % Ensure positive definiteness
        %         try
        %             obj.locals(i).P = inv(new_info);
        %             obj.locals(i).x = obj.locals(i).P * new_info_vec;
        %         catch
        %             % Fallback to simple weighted combination if numerical issues
        %             weight = 0.1; % Small weight for stability
        %             obj.locals(i).x = (1-weight) * x_local + weight * obj.x;
        %             obj.locals(i).P = (1-weight) * P_local + weight * obj.P;
        %         end
        %     end
        % end
        
        % --- Proper ISF calculation ---
        function finalizeCalibration(obj)
            fprintf('--- CALIBRATION COMPLETE ---\n');
            
            obj.calculateISFs();
            
            % Reset Covariance matrices and Q to correct values

            % Switch to operational mode
            obj.mode = 'operational';
            obj.operational_steps = 0;
            obj.operational_errors = zeros(1, numel(obj.locals));
            
            fprintf('Switching to OPERATIONAL mode.\n');
            fprintf('Learned Information Sharing Factors (ISF):\n');
            for i = 1:numel(obj.ISF)
                fprintf('  Local Filter %d: %.4f\n', i, obj.ISF(i));
            end
            fprintf('ISF sum: %.4f\n', sum(obj.ISF));
            fprintf('--------------------------------\n');
        end
        
        % --- Correct ISF calculation ---
        function calculateISFs(obj)
            mse = max(obj.calibration_errors / obj.calibration_steps, 1e-10);
            inverse_mse = 1 ./ mse;
            
            % ISF as direct weights (sum = 1)
            obj.ISF = inverse_mse / sum(inverse_mse);
            
            % Apply bounds
            obj.ISF = max(obj.ISF, 0.05);  % minimum 5%
            obj.ISF = obj.ISF / sum(obj.ISF);  % renormalize
        end

        % --- Operational step without reference filter ---
        function operationalStep(obj, z_cell, z_ref, fuseFlag)
            % Note: z_ref is ignored in operational mode
            
            % 1) Predict local filters only
            for i = 1:numel(obj.locals)
                obj.locals(i).predict();
            end
            
            % 2) Fault detection and local updates
            for i = 1:numel(obj.locals)
                zi = z_cell{i};
                if all(~isnan(zi))
                    % Perform fault detection before update
                    obj.detectFault(i, zi);
                    
                    % Only update if no fault detected
                    if obj.fault_flags(i) == 1
                        obj.locals(i).update(zi);
                    else
                        obj.handleFault(i);
                    end
                end
            end
            
            obj.operational_steps = obj.operational_steps + 1;
            
            % 3) Fusion (local filters only)
            if fuseFlag
                obj.fuseFilters();
            end
        end
        
        % --- Centralized fault handling ---
        function handleFault(obj, i)
            switch obj.method
                case 1
                    fprintf('Fault detected in local filter %d - assigning last global value\n', i);
                    obj.locals(i).x = obj.x;
                case 2
                    fprintf('Fault detected in local filter %d - skipping update\n', i);
                case 3
                    fprintf('Fault detected in local filter %d - assigning last global value and increasing covariance\n', i);
                    obj.locals(i).x = obj.x;
                    obj.locals(i).P = obj.P * 2;  % Less aggressive covariance inflation
            end
        end
        
        function detectFault(obj, local_idx, measurement)
            % --- Improved fault detection with clear precedence ---
            
            local_filter = obj.locals(local_idx);
            
            % Get measurement model parameters
            H = local_filter.sensor.H;
            R = local_filter.sensor.R;
            x_pred = local_filter.x;
            P_pred = local_filter.P;
            
            % Predicted measurement and residual
            z_pred = H * x_pred;
            residual = measurement - z_pred;
            residual_covar = H * P_pred * H' + R;
            
            % Primary fault detection: Chi-square test
            lambda = residual' / residual_covar * residual;
            chi_square_fault = lambda >= obj.chi_square_threshold(local_idx);
            
            % Update sliding window
            obj.updateSlidingWindow(local_idx, residual, residual_covar);
            
            % Secondary fault detection: Sliding window test (if enough data)
            sliding_window_fault = false;
            if length(obj.residual_history{local_idx}) >= obj.window_size
                sliding_window_fault = obj.slidingWindowTest(local_idx);
            end
            
            % ---  Clear fault detection logic ---
            if chi_square_fault || sliding_window_fault
                obj.fault_flags(local_idx) = 0;  % fault detected
            else
                obj.fault_flags(local_idx) = 1;  % normal operation
            end
        end
        
        function updateSlidingWindow(obj, local_idx, residual, theoretical_covar)
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
        
        % --- Sliding window test with clear return value ---
        function fault_detected = slidingWindowTest(obj, local_idx)
            history = obj.residual_history{local_idx};
            covar_history = obj.covar_history{local_idx};
            
            if length(history) < obj.window_size
                fault_detected = false;
                return;
            end
            
            % Calculate actual and theoretical covariances
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
            
            % ---  Complete coverage of eta range ---
            if eta > 2.5 || eta < 0.4
                fault_detected = true;
            else
                fault_detected = false;
            end
        end
        
        function status = getFaultStatus(obj)
            status = obj.fault_flags;
        end
        
        function printFaultStatus(obj)
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