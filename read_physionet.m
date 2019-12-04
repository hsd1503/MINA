function read_physionet()

%%%%%%%%%%%%%%%%%%%%%%%%%
% data format sample: 
% A000001,N,1,1,1,1,1
%%%%%%%%%%%%%%%%%%%%%%%%%


fin = fopen('REFERENCE.csv');
str=fgetl(fin);
fout = fopen('data/long.csv','w');

while ischar(str)
    line=textscan(str,'%s');
    tmp = strsplit(line{1}{1}, ',');
    pid = tmp{1};
    label = tmp{2};
    
    [tm,ecg,fs,siginfo]=rdmat(strcat('training2017/', pid));

    tmp_len = length(ecg);
    fprintf(fout, '%s,', pid);
    fprintf(fout, '%s,', label);
    fprintf(fout, '%f,',ecg(1:tmp_len-1));
    fprintf(fout, '%f\n',ecg(tmp_len));
    str=fgetl(fin);
end
fclose(fout);
