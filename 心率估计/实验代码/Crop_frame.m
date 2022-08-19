% ����������Ƶ����ȡ����Ȥ�������գ�
function frame = Crop_frame(frame_raw, k, ROI_x, ROI_y)
    % frame_raw ������Ƶ֡
    % ROI_x ����Ȥ����� landmark �ĺ�����
    % ROI_y ����Ȥ����� landmark ��������
    % frame �ü����Ĳ���
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
