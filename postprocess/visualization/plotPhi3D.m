clear all
close all

phi = loadMETA('../3d_results_using_vectorMomentum/phi.mhd');

grid_x = reshape(phi(1, :, :, :), size(phi, 2), size(phi, 3), size(phi, 4));
grid_y = reshape(phi(2, :, :, :), size(phi, 2), size(phi, 3), size(phi, 4));
grid_z = reshape(phi(3, :, :, :), size(phi, 2), size(phi, 3), size(phi, 4));

offset = 40;
stepsize = 3;

figure, hold on
for i = 1+offset:stepsize:size(grid_x, 1)-offset
    for j = 1+offset:stepsize:size(grid_x, 2)-offset
        x = grid_x(i, j, 1+offset:end-offset);
        y = grid_y(i, j, 1+offset:end-offset);
        z = grid_z(i, j, 1+offset:end-offset);
        plot3(x(:), y(:), z(:), 'k.', 'LineWidth', 1);
    end
end
for i = 1+offset:stepsize:size(grid_x, 1)-offset
    for k = 1+offset:stepsize:size(grid_x, 3)-offset
        x = grid_x(i, 1+offset:end-offset, k);
        y = grid_y(i, 1+offset:end-offset, k);
        z = grid_z(i, 1+offset:end-offset, k);
        plot3(x(:), y(:), z(:), 'k.', 'LineWidth', 1);
    end
end
for j = 1+offset:stepsize:size(grid_x, 2)-offset
    for k = 1+offset:stepsize:size(grid_x, 3)-offset
        x = grid_x(1+offset:end-offset, j, k);
        y = grid_y(1+offset:end-offset, j, k);
        z = grid_z(1+offset:end-offset, j, k);
        plot3(x(:), y(:), z(:), 'k.', 'LineWidth', 1);
    end
end
hold off
axis equal
axis off