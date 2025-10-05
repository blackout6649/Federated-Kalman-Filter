%% Traditional FKF class (for comparison)
classdef FederatedKFTraditional < handle
    properties
        locals, weight, x, P, name
    end
    methods
        function obj = FederatedKFTraditional(locals, weight, name)
            if nargin < 2, weight = []; end
            if nargin < 3, name = "FKF_Traditional"; end
            obj.locals = locals; obj.weight = weight; obj.name = name;
            [x0, P0] = obj.locals(1).estimate();
            obj.x = x0; obj.P = P0;
        end
        function step(obj, z_cell, fuseFlag)
            N = length(z_cell);
            % Traditional federated filter without fault detection
            for i = 1:N
                obj.locals(i).predict();
            end
            for i = 1:N
                zi = z_cell{i};
                if all(~isnan(zi))
                    obj.locals(i).update(zi);  % No fault detection
                end
            end
            if fuseFlag
                X = cell(1,numel(obj.locals)); P = cell(1,numel(obj.locals));
                for i = 1:numel(obj.locals)
                    [xi, Pi] = obj.locals(i).estimate();
                    X{i} = xi; P{i} = Pi;
                end
                [obj.x, obj.P] = FusionCenter.fuse(X, P);
                for i = 1:numel(obj.locals)
                    obj.locals(i).x = obj.x;
                    obj.locals(i).P = obj.P * obj.weight;
                end
            end
        end
    end
end