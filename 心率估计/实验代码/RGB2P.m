% 通过 RGB 的颜⾊变化，获得⽣理信号的表示
% 颜色空间投影
function P = RGB2P(R_value, G_value, B_value, Method, Window)
    global numFrames;
    % R_value R 通道颜⾊变化
    % G_value G 通道颜⾊变化
    % B_value B 通道颜⾊变化
    % Method 数据平滑滤波的⽅法，本次实验⼀律采⽤ Gaussian 滤波
    % Window 平滑滤波的窗⼝⼤⼩
    % P ⽣理信号
    Smooth_R_value = smoothdata(R_value, Method, Window);
    Smooth_G_value = smoothdata(G_value, Method, Window);
    Smooth_B_value = smoothdata(B_value, Method, Window);

    Avg_Smooth_R_value = sum(Smooth_R_value) / numFrames;
    Avg_Smooth_G_value = sum(Smooth_G_value) / numFrames;
    Avg_Smooth_B_value = sum(Smooth_B_value) / numFrames;

    Normal_R_value = Smooth_R_value ./ Avg_Smooth_R_value;
    Normal_G_value = Smooth_G_value ./ Avg_Smooth_G_value;
    Normal_B_value = Smooth_B_value ./ Avg_Smooth_B_value;

    Avg_Normal_R_value = sum(Normal_R_value) / numFrames;
    Avg_Normal_G_value = sum(Normal_G_value) / numFrames;
    Avg_Normal_B_value = sum(Normal_B_value) / numFrames;

    S1 = 3 .* Normal_R_value - 2 .* Avg_Normal_G_value;
    S2 = 1.5 .* Normal_R_value + Avg_Normal_G_value - 1.5 .* Normal_B_value;

    Var_S1 = var(S1, 1);
    Var_S2 = var(S2, 1);
    alp = Var_S1 / Var_S2;

    P = S1 - alp * S2;
end
