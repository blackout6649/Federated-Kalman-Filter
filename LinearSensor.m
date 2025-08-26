classdef LinearSensor
    % Linear measurement: z = Hx + V, v ~ N(0, R)
    properties
        H
        R
        name
    end 
    methods 
        function obj = LinearSensor(H, R ,name)
            if nargin < 3, name = "LinearSensor"; end
            obj.H = H; obj.R = R; obj.name = name;
        end 
        function z = measure(obj, xRef)
            m = size(obj.H, 1);
            z = obj.H * xRef + mvnrnd(zeros(m,1), obj.R)'; % Create appropriately noisey measurements 
        end 
    end 
end