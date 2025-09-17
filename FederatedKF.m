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
        
        % --- MODIFIED: Added MSE property to reset after fusion ---
        mse                  % Mean Squared Error for each local filter

        % --- NEW: Reference weight property ---
        reference_weight     % Manual weight of the reference filter (e.g., 0.6)
    end
    
    methods
        % --- MODIFIED: Constructor to accept reference_weight ---
        function obj = FederatedKF(locals, reference_filter, calibration_duration, name, confidence_level, window_size, method, reference_weight)
            if nargin < 4, name = "CalibratingFKF"; end
            if nargin < 5, confidence_level = 0.05; end
            if nargin < 6, window_size = 10; end
            if nargin < 7, method = 1; end
            % --- NEW: Set default reference weight ---
            if nargin < 8, reference_weight = 0.6; end
            
            obj.locals = locals;
            obj.name = name;
            obj.confidence_level = confidence_level;
            obj.window_size = window_size;
            obj.method = method;
            obj.reference_weight = reference_weight; % Store the new property
            
            num_locals = numel(obj.locals);
            
            % --- NEW: Initialize Calibration ---
            obj.mode = 'calibration'; % Start in calibration mode
            obj.reference_filter = reference_filter;
            obj.calibration_duration = calibration_duration;
            obj.calibration_steps = 0;
            obj.calibration_errors = zeros(1, num_locals);
            % --- MODIFIED: Initialize MSE and ISF arrays ---
            obj.mse = zeros(1, num_locals); 
            obj.ISF = ones(1, num_locals);
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
        
        % --- MODIFIED: Main step function ---
        function step(obj, z_cell, z_ref, fuseFlag)
            % z_cell: cell{N} of measurements for local filters
            % z_ref: measurement for the reference filter
            
            if strcmp(obj.mode, 'calibration')
                obj.calibrationStep(z_cell, z_ref);
                if fuseFlag
                    obj.fuseFilters();
                end
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
        
        % --- NEW: Centralized fusion function with learned weights ---
        function fuseFilters(obj)
            fprintf('Fusing filters...\n');
            
            % Recalculate ISFs before each fusion
            obj.calculateISFs();
            
            valid_indices = find(obj.fault_flags == 1);
            if isempty(valid_indices), warning('All local filters have faults - skipping fusion'); return; end
            
            num_valid = numel(valid_indices);
            
            % --- MODIFIED: Include reference filter in the fusion ---
            X = cell(1, num_valid + 1); 
            P = cell(1, num_valid + 1);
            ISF_valid = zeros(1, num_valid + 1);
            
            % Add reference filter to the fusion set
            [X{1}, P{1}] = obj.reference_filter.estimate();
            % --- MODIFIED: Use the new reference_weight property ---
            ISF_valid(1) = obj.reference_weight; 
            
            % Add local filters to the fusion set
            for j = 1:num_valid
                local_idx = valid_indices(j);
                [X{j+1}, P{j+1}] = obj.locals(local_idx).estimate();
                % Calculate the weight for this filter based on its ISF
                ISF_valid(j+1) = obj.ISF(local_idx);
            end
            
            % Normalize the remaining weights so they sum to (1 - reference_weight)
            sum_of_other_ISF = sum(ISF_valid(2:end));
            if sum_of_other_ISF > 1e-6
                ISF_valid(2:end) = ISF_valid(2:end) * (1 - obj.reference_weight) / sum_of_other_ISF;
            else
                % All other filters have zero weight, distribute the remaining weight equally
                ISF_valid(2:end) = (1 - obj.reference_weight) / num_valid;
            end
            
            % Perform fusion using the new weights
            [obj.x, obj.P] = FusionCenter.fuse_with_weights(X, P, ISF_valid);
            
            % --- NEW: Reset each local filter's state and covariance
            N = numel(obj.locals);
            for i = 1:N
                obj.locals(i).x = obj.x;
                % Reset the covariance based on the new fused covariance
                % if obj.ISF(i) > 1e-6
                %     obj.locals(i).P = obj.P * N / obj.ISF(i);
                % else
                %     obj.locals(i).P = obj.P * 1e12; 
                % end
                obj.locals(i).P = obj.P * N;
            end
            
            % --- NEW: Reset MSEs after fusion
            obj.calibration_errors = zeros(1, numel(obj.locals));
            obj.calibration_steps = 0; % Reset the counter
        end
        
        % --- NEW: Finalize calibration and calculate ISFs ---
        function finalizeCalibration(obj)
            fprintf('--- CALIBRATION COMPLETE ---\n');
            
            obj.calculateISFs();
            
            % Switch to operational mode
            obj.mode = 'operational';
            
            fprintf('Switching to OPERATIONAL mode.\n');
            fprintf('Learned Information Sharing Factors (ISF):\n');
            for i = 1:numel(obj.ISF)
                fprintf('  Local Filter %d: %.4f\n', i, obj.ISF(i));
            end
            fprintf('--------------------------------\n');
        end
        
        % --- NEW: Helper function to calculate ISFs
        function calculateISFs(obj)
            % Calculate Mean Squared Error (MSE) for each local filter
            obj.mse = obj.calibration_errors / obj.calibration_steps;
            
            % Calculate ISFs based on inverse MSE
            inverse_mse = 1 ./ obj.mse;
            sum_inverse_mse = sum(inverse_mse);
            
            beta = inverse_mse / sum_inverse_mse;
            
            % According to FKF, the ISF (alpha) should be N*beta
            obj.ISF = numel(obj.locals) * beta;
        end
        
        % --- RENAMED: from 'step' to 'operationalStep' ---
        function operationalStep(obj, z_cell, fuseFlag)
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
            obj.fuseFilters();
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