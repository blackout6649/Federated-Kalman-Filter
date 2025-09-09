classdef LinearSensor
    % Linear measurement: z = Hx + v, v ~ N(0, R)
    properties
        H
        R
        measInterval
        name
        faultProb      % probability of fault per measurement
        faultMagnitude % scale of fault (offset or noise inflation)
    end
    methods
        function obj = LinearSensor(H, R, measInterval, name, faultProb, faultMagnitude)
            if nargin < 4, name = "LinearSensor"; end
            if nargin < 5, faultProb = 0; end
            if nargin < 6, faultMagnitude = 50; end
            obj.H = H; 
            obj.R = R; 
            obj.measInterval = measInterval; 
            obj.name = name;
            obj.faultProb = faultProb;
            obj.faultMagnitude = faultMagnitude;
        end

        function z = measure(obj, xtrue, k)
            if mod(k, obj.measInterval) == 0
                % Normal measurement
                v = mvnrnd(zeros(size(obj.R,1),1), obj.R)';
                z = obj.H * xtrue + v;

                % Introduce a fault with probability faultProb
                if rand < obj.faultProb
                    z = z + obj.faultMagnitude * (2*rand(size(z)) - 1);
                end
            else
                z = [NaN; NaN]; % no measurement
            end
        end
    end
end
