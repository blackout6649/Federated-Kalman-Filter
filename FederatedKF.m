classdef FederatedKF < handle
    % Orchestrates multiple LocalKalmanFilter objects + fusion center
    properties
        locals   % array of LocalKalmanFilter
        weight  % fusion weights (optional)
        x        % fused state
        P        % fused covariance
        name
    end
    methods
        function obj = FederatedKF(locals, weight, name)
            if nargin < 2, weight = []; end
            if nargin < 3, name = "FKF"; end
            obj.locals = locals; obj.weight = weight; obj.name = name;
            % Initialize fused with first local
            [x0, P0] = obj.locals(1).estimate();
            obj.x = x0; obj.P = P0;
        end
        function step(obj, z_cell, fuseFlag)
            % z_cell: cell{N} of measurements for each local (or [] to skip)
            % 1) local predicts
            for i = 1:numel(obj.locals)
                obj.locals(i).predict();
            end
            % 2) local updates (skip if measurement missing)
            for i = 1:numel(obj.locals)
                zi = z_cell{i};
                if ~isempty(zi)
                    obj.locals(i).update(zi);
                end
            end
            % 3) fusion (skip if haven't reached fusion step)
            if fuseFlag
                X = cell(1,numel(obj.locals));
                P = cell(1,numel(obj.locals));
                for i = 1:numel(obj.locals)
                    [xi, Pi] = obj.locals(i).estimate();
                    X{i} = xi; P{i} = Pi;
                end
                [obj.x, obj.P] = FusionCenter.fuse(X, P);
                % Reset 
                for i = 1:numel(obj.locals)
                    obj.locals(i).x = obj.x;
                    obj.locals(i).P = obj.P * obj.weight;
                end
            end 
        end
    end
end
