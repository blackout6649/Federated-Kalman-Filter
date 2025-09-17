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
        
        % --- NEW: Calibration Properties ---
        mode                 % 'calibration' or 'operational'
        reference_filter     % The trusted "System B" filter
        calibration_duration % Number of steps for the calibration phase
        calibration_steps    % Counter for the current calibration step
        calibration_errors   % Stores accumulated squared errors for each local filter
    end
    
    methods
        % --- MODIFIED: Constructor ---
        function obj = FederatedKF(locals, reference_filter, calibration_duration, name, confidence_level, window_size, method)
            if nargin < 4, name = "CalibratingFKF"; end
            if nargin < 5, confidence_level = 0.05; end
            if nargin < 6, window_size = 10; end
            if nargin < 7, method = 1; end
            
            obj.locals = locals;
            obj.name = name;
            obj.confidence_level = confidence_level;
            obj.window_size = window_size;
            obj.method = method;
            
            num_locals = numel(obj.locals);
            
            % --- NEW: Initialize Calibration ---
            obj.mode = 'calibration'; % Start in calibration mode
            obj.reference_filter = reference_filter;
            obj.calibration_duration = calibration_duration;
            obj.calibration_steps = 0;
            obj.calibration_errors = zeros(1, num_locals);
            fprintf('--- FKF "%s" initialized in CALIBRATION mode for %d steps. ---\n', obj.name, obj.calibration_duration);
            
            % Initialize ISF with equal weighting (will be overwritten after calibration)
            obj.ISF = ones(1, num_locals) * num_locals;

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
            % z_ref: measurement for the reference filter (only used in calibration)
            
            if strcmp(obj.mode, 'calibration')
                obj.calibrationStep(z_cell, z_ref);
            else % operational mode
                obj.operationalStep(z_cell, fuseFlag);
            end
       end

        % --- NEW: Step function for the calibration phase ---
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
                % Accumulate the squared norm of the error
                obj.calibration_errors(i) = obj.calibration_errors(i) + (error_vec' * error_vec);
            end
            
            obj.calibration_steps = obj.calibration_steps + 1;
            fprintf('Calibration step %d/%d...\n', obj.calibration_steps, obj.calibration_duration);
            
            % Check if calibration is complete
            if obj.calibration_steps >= obj.calibration_duration
                obj.finalizeCalibration();
            end
        end
        
        % --- NEW: Finalize calibration and calculate ISFs ---
        function finalizeCalibration(obj)
            fprintf('--- CALIBRATION COMPLETE ---\n');
            
            % Calculate Mean Squared Error (MSE) for each local filter
            mse = obj.calibration_errors / obj.calibration_steps;
            
            % Calculate ISFs based on inverse MSE
            inverse_mse = 1 ./ mse;
            sum_inverse_mse = sum(inverse_mse);
            
            beta = inverse_mse / sum_inverse_mse;
            
            % According to FKF, the ISF (alpha) should be N*beta
            obj.ISF = numel(obj.locals) * beta;
            
            % Switch to operational mode
            obj.mode = 'operational';
            
            fprintf('Switching to OPERATIONAL mode.\n');
            fprintf('Learned Information Sharing Factors (ISF):\n');
            for i = 1:numel(obj.ISF)
                fprintf('  Local Filter %d: %.4f\n', i, obj.ISF(i));
            end
            fprintf('--------------------------------\n');
        end
        
        % --- RENAMED: from 'step' to 'operationalStep' ---
        function operationalStep(obj, z_cell, fuseFlag)
            % This function contains the original logic from your step method
            
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
            
            % 3) fusion
            if fuseFlag
                valid_indices = find(obj.fault_flags == 1);
                if isempty(valid_indices), warning('All local filters have faults - skipping fusion'); return; end
                
                num_valid = numel(valid_indices);
                X = cell(1, num_valid); P = cell(1, num_valid);
                for j = 1:num_valid
                    local_idx = valid_indices(j);
                    [X{j}, P{j}] = obj.locals(local_idx).estimate();
                end
                
                [obj.x, obj.P] = FusionCenter.fuse(X, P);
                
                % --- MODIFIED: Reset with learned ISFs ---
                % The standard FKF reset is P_i = P_global / beta_i
                % Since our ISF = N * beta_i, then beta_i = ISF_i / N
                % So, P_i = P_global * N / ISF_i
                N = numel(obj.locals);
                for i = 1:N
                    obj.locals(i).x = obj.x;
                    if obj.ISF(i) > 1e-6 % Avoid division by zero
                        obj.locals(i).P = obj.P * N / obj.ISF(i);
                    else
                        % Handle case of a sensor with zero trust
                        obj.locals(i).P = obj.P * 1e12; % Assign very high uncertainty
                    end
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