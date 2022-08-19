%�Ӹ���Ȥ��������ȡ�� RGB 3 ͨ�����Ձ9�0�仯
%����������Ȥ������Ձ9�0ֵ��ƽ�����Դﵽ����ā9�0��
function RGB_Record = Get_RGB(frame)
    % frame �����ü������Ƶ֡
    % RGB_Record ���õ��� RGB �Ձ9�0ֵ
    [Row, Col, Dep] = size(frame);
    pixel_count = 0;

    R_value = 0.0;
    G_value = 0.0;
    B_value = 0.0;

    for i = 1:Row

        for j = 1:Col

            if frame(i, j, 1) == 0 && frame(i, j, 2) == 0 && frame(i, j, 3) == 0
                continue;
            else
                pixel_count = pixel_count + 1;
                R_value = R_value + double(frame(i, j, 1));
                G_value = R_value + double(frame(i, j, 2));
                B_value = R_value + double(frame(i, j, 3));

            end

        end

    end

    Avg_R_value = R_value / pixel_count;
    Avg_G_value = G_value / pixel_count;
    Avg_B_value = B_value / pixel_count;

    RGB_Record = [Avg_R_value Avg_G_value Avg_B_value];
end
