classdef CentralizedKF < handle
    % Centralized linear Kalman filter with asynchronous multi-sensor updates
    properties
        model      % MotionModel with F,G,Q
        sensors    % cell array of LinearSensor (for H,R access)
        x          % state estimate
        P          % covariance
        name
    end
    methods
        function obj = CentralizedKF(model, x0, P0, sensors, name)
            if nargin < 5, name = "CKF"; end
            obj.model   = model;
            obj.sensors = sensors;
            obj.x       = x0;
            obj.P       = P0;
            obj.name    = name;
        end

        function predict(obj)
            F = obj.model.F;
            G = obj.model.G;
            Q = obj.model.Q;
            obj.x = F * obj.x;
            obj.P = F * obj.P * F' + G * Q * G';
        end

        function update(obj, z_cell)
            % z_cell: cell{Ns} of measurements (some may be [])
            % Build stacked H, z, and block-diag R from available sensors
            Hc = [];
            zc = [];
            Rc_blocks = {};
            for i = 1:numel(z_cell)
                zi = z_cell{i};
                if all(~isnan(zi))
                    Hi = obj.sensors{i}.H;
                    Ri = obj.sensors{i}.R;
                    Hc = [Hc; Hi];           
                    zc = [zc; zi];           
                    Rc_blocks{end+1} = Ri;   
                end
            end
            if isempty(Hc)
                return; % no measurements this step
            end
            Rc = blkdiag(Rc_blocks{:});

            % Standard KF update with stacked measurement
            S = Hc * obj.P * Hc' + Rc;
            K = obj.P * Hc' / S;
            innov = zc - Hc * obj.x;
            obj.x = obj.x + K * innov;
            I = eye(size(obj.P));
            obj.P = (I - K * Hc) * obj.P;
        end

        function step(obj, z_cell)
            obj.predict();
            obj.update(z_cell);
        end
    end
end
