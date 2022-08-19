% 输入人脸视频，获取感兴趣区域（脸颊）
function frame = Crop_frame(frame_raw, k, ROI_x, ROI_y)
    % frame_raw 输入视频帧
    % ROI_x 感兴趣区域的 landmark 的横坐标
    % ROI_y 感兴趣区域的 landmark 的纵坐标
    % frame 裁剪出的部分
    global lmk;

    frame_raw_R = frame_raw(:, :, 1);
    frame_raw_G = frame_raw(:, :, 2);
    frame_raw_B = frame_raw(:, :, 3);

    intrest_region_x = lmk(k, ROI_x);
    intrest_region_y = lmk(k, ROI_y);

    bw = roipoly(frame_raw, intrest_region_x, intrest_region_y);

    frame_R = frame_raw_R .* double(bw);
    frame_G = frame_raw_G .* double(bw);
    frame_B = frame_raw_B .* double(bw);

    frame(:, :, 1) = frame_R;
    frame(:, :, 2) = frame_G;
    frame(:, :, 3) = frame_B;

end
