function [true_data, z] = generate_data_2DNEW(Q, R)

% generate_data_2D.m
% This function generates the data for a constant-velocity 2D motion example.
% The motion is divided in 5 segments, passing through the points: 
%  - (200,-100)  to (100,100)
%  - (100,100)   to (100,300)
%  - (100,300)   to (-200,300)
%  - (-200,300)  to (-200,-200)
%  - (-200,-200) to (0,0)

% Q are the elements of the process noise diagonal covariance matrix (only for position)
% R are the elements of the measurement noise diagonal covariance matrix. The angle noise should be 1e+6 smaller than the noise for r
% z is the [distance; orientation] data measured from the origin
% true_data are the true values of the position and speed

% example of use
% [true_data, z] = generate_data_2D([10 10], [100, 1e-3]);

plot_grafs = true;
nSegments = 5;
points = [200, -100; 100, 100; 100, 300; -200, 300; -200, -200; 0,0];
points = points(1:nSegments+1, :);

dp = diff(points);
dist = (dp).^2; 
dist = round(sqrt(dist(:,1) + dist(:,2))); % distance
ang = atan2(dp(:, 2), dp(:, 1)); % orientation

NumberOfDataPoints = sum(dist);

T = 0.5; %[s]

v_set = 2 * [cos(ang) sin(ang)];%[-0.4472, 0.8944; 0, 1; -1, 0; 0, -1; sqrt(2)/2, sqrt(2)/2]; % 2 m/s

v = []; % velocity vector of constant magnitude 
for idx = 1:nSegments
    v = [v; (repmat(v_set(idx, :), dist(idx), 1))];
end


% ==motion generation====================================================

A = [1 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 0];
B = [T 0; 1 0; 0 T;0 1];
G = [T.^2/2 0; T 0; 0 T.^2/2; 0 T]; 

if (length(Q) == 1), Q = [Q; Q]; end
w_x = sqrt(Q(1)) * randn(NumberOfDataPoints, 1); %acceleration noise in x-direction
w_y = sqrt(Q(2)) * randn(NumberOfDataPoints, 1); %acceleration noise in y-direction
w = [w_x w_y];

x(1, :) = [200 0 -100 0];
for idx = 2:NumberOfDataPoints
    x(idx, :) = ( A * x(idx-1, :)' + B * v(idx,:)' + G * w(idx, :)' )';
end

if plot_grafs, 
    figure(1);
    plot(x(:, 1), x(:, 3), '.'); 
    title('trajectory');
    xlabel('x-axis [m]'); ylabel('y-axis [m]');
end

true_data = x; % 2D data: [px; vx; py; vy]

% ==measurement generation===============================================
position = x(:, [1,3]); % 2D position data

% distance and orientation with respect to the origin
z = zeros(NumberOfDataPoints, 2);
for idx = 1:NumberOfDataPoints
    z(idx,1) = sqrt(position(idx, :) * position(idx, :)');
    z(idx,2) = atan2(position(idx, 2), position(idx, 1)); 
end

% unwrap radian phases by changing absolute jumps greater than pi to their 2*pi complement
z(:, 2) = unwrap(z(:, 2));

if (length(R) == 1), R = [R; R]; end

v_meas = [sqrt(R(1)) * randn(NumberOfDataPoints, 1), sqrt(R(2)) * randn(NumberOfDataPoints, 1)];
z_exact = z;
z = z + v_meas; % add measurement noise

if plot_grafs
    figure(2); xlab ={' ', 'Time step [s]'}; ylab ={'r [m]', '\theta [rad]'};
    for idx = 1:2
        subplot(2,1,idx); hold on;
        plot(z(:, idx), 'xr'); plot(z_exact(:, idx), 'b'); 
        legend('Measured (noisy)', 'Exact', 'Box', 'Off', 'Location', 'South', 'Orientation', 'Horizontal');
        xlabel(xlab{idx}); ylabel(ylab{idx}); 
        set(gca, 'Box', 'Off', 'FontSize', 12);
    end
end
    
% save measurements.mat z 
% save trueDataLab3.mat true_data;