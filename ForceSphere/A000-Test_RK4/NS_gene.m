% 设置矩阵的大小
i = 10000; % 行数
j = 3; % 列数

% 生成随机矩阵
matrix = [rand(i, 2)*3, zeros(i, 1)];

% 保存矩阵为文本文件
filename = 'NS_10000.txt';
dlmwrite(filename, matrix, 'delimiter', '\t', 'precision', '%.3f');