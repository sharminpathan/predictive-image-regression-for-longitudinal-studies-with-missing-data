clear all
%close all

addpath('../oasis_2d_samples');
phi = loadMETA('../oasis_2d_samples/phi.mhd');
image = loadMETA('../oasis_2d_samples/I1.mhd');

phi = loadMETA('../atlas_nondem_(demented_model)/m4_phi.mhd');
image = loadMETA('../atlas_nondem_(demented_model)/I-4_hat.mhd');

%phi = loadMETA('../testing/m11_phi.mhd');
%image = loadMETA('../testing/I11_hat.mhd');

grid_x = reshape(phi(1, :, :), size(phi, 2), size(phi, 3));
grid_y = reshape(phi(2, :, :), size(phi, 2), size(phi, 3));

figure, hold on
for i = 1:size(grid_x, 1)
    plot(grid_x(i, :)+1, grid_y(i, :)+1, 'Color', [0.35 0.35 0.35], 'LineWidth', 1);
end
for j = 1:size(grid_x, 2)
    plot(grid_x(:, j)+1, grid_y(:, j)+1, 'Color', [0.35 0.35 0.35], 'LineWidth', 1);
end
hold off
axis equal
axis off
camroll(180)

figure, imshow(image); colormap(gray);
hold on
for i = 1:size(grid_x, 1)
    plot(grid_x(i, :)+1, grid_y(i, :)+1, 'Color', [0.5 0.5 1.0], 'LineWidth', 1);
end
for j = 1:size(grid_x, 2)
    plot(grid_x(:, j)+1, grid_y(:, j)+1, 'Color', [0.5 0.5 1.0], 'LineWidth', 1);
end
hold off
axis equal
axis off