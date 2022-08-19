% ͨ���9�1���źţ����9�2��
% ʱ��ռ�ķ�����
function [Golden_HR, HR] = Check(Path, Window)

    global ROI_forehead_x ROI_forehead_y;
    global ROI_left_cheek_x ROI_left_cheek_y;

    ROI_forehead_x = [28 22 21 20 24 23];
    ROI_forehead_y = ROI_forehead_x + 68;

    ROI_left_cheek_x = [2 3 4 5 32 41];
    ROI_left_cheek_y = ROI_left_cheek_x + 68;

    ROI_right_cheek_x = [13 14 15 16 17 36];
    ROI_right_cheek_y = ROI_right_cheek_x + 68;

    ROI_full_cheek_x = [2 3 4 5 32 31 36 13 14 15 16 47 29 41];
    ROI_full_cheek_y = ROI_full_cheek_x + 68;

    golden_hr_path = strcat(Path, '/HR_gt.txt');
    Golden_HR = textread(golden_hr_path);

    fps_path = strcat(Path, '/fps.txt');
    FPS = textread(fps_path);

    global lmk
    lmk_path = strcat(Path, '/lmk.csv');
    lmk = csvread(lmk_path, 1, 5);

    % ������Ƶλ��
    video_path = strcat(Path, '/video.avi');
    obj = VideoReader(video_path);

    % ֡������
    global numFrames;
    numFrames = obj.NumberOfFrames;

    % ��ȡÿ��֡
    for k = 1:numFrames
        % fprintf("Processing The %dth Frame\n", k);

        %��ȡ��k֡
        frame_raw = double(read(obj, k));
        frame_full_cheek = Crop_frame(frame_raw, k, ROI_full_cheek_x, ROI_full_cheek_y);
        RGB_Record_full_cheek = Get_RGB(frame_full_cheek);
        R_value(k) = RGB_Record_full_cheek(1);
        G_value(k) = RGB_Record_full_cheek(2);
        B_value(k) = RGB_Record_full_cheek(3);

    end

    P = RGB2P(R_value, G_value, B_value, 'gaussian', Window);
    % �Լ��9�8ֵ��Ϊ�9�2�����ڵĲ��壬���9�2��
    [Pks, Locs] = findpeaks(P);
    numLocalMax = size(Locs);

    for i = 1:numLocalMax(2) - 1
        Distance(i) = Locs(i + 1) - Locs(i);
    end

    Avg_Distance = sum(Distance) / (numLocalMax(2) - 1);
    HR = uint8(FPS / Avg_Distance * 60);

end
