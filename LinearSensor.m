classdef LinearSensor
    % Linear measurement: z = Hx + V, v ~ N(0, R)
    properties
        H
        R
        name
        meas_flag
    end 
    methods 
        function obj = LinearSensor(H, R, meas_flag, name)
            if nargin < 3, name = "LinearSensor"; end
            obj.H = H; obj.R = R; obj.name = name; obj.meas_flag = meas_flag;
        end 
        function z = measure(obj, xRef, k)
            if mod(k, obj.meas_flag) == 0
                m = size(obj.H,1);
                z = obj.H * xRef + mvnrnd(zeros(m,1), obj.R)';
            else
                z = [];
            end
        end 
    end 
end