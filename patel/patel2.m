tic
tmp=data1;%改 
[m, n] = size(tmp);
c =zeros(m,n);
%M是指生成在patel生成的的patelsim10050的行数，其实这个无所谓，只要比4950大，能把c的行存进去就行；N+2是指生成的patelsim10050的列数，只要比5大，能把c的列存进去就行
count =0;
for i = 1 : n-1
    for j = i+1 : n % 逐行打印出来
        count=count+1;
        a=tmp(:,[i j]);
        b=Pate(a);           
         c(count,1)=i;
         c(count,2)=j;
         c(count,3:5)=b; 
         fprintf(fid,'%d\t',c)
         fprintf(fid, '\n');
     end
   
end
toc
c_New=c;
[cnew_m,cnew_n]=size(c_New);
%卡阈值。设置阈值，当第五列的值小于s时，去掉这条边，从而保证输出的边的条数不要太多，应该与groundtruth网络的边数大致相同
[~,I]=sort(-c_New(:,3));%把c_New按照降序排列
c_New=c_New(I,:);
csave=c_New(1:6,:); %改阈值
Mat2txt('E:\投稿\综述\英文综述实验及结果\Patel\40\1\patel_1.txt',csave)
[m2,n2]=size(csave);
 h=load('E:\投稿\中文投稿\对比算法代码\数据\h1.txt'); %改标准网络
 [h_m,h_n]=size(h);
%当标准网络超过所取得边数时
% csave_1=zeros(h_m-m2,n2);
%  csave_2=[csave;csave_1];
%当所取得边数超过标准网络时
h_1=zeros(m2-h_m,h_n);
h_2=[h;h_1];
out=[];
for i3=1:m2
    for j3=1:m2
 
%    if   csave_2(i3,1)==h(j3,1)&&csave_2(i3,2)==h(j3,2)    %对比生成的patel结果矩阵中与groudtruth相同的边
%      out=[out;csave_2(i3,:) h(j3,:)]  %输出patel结果和grountrueth相同的边组成的矩阵
   if csave(i3,1)==h_2(j3,1)&&csave(i3,2)==h_2(j3,2)    %对比生成的patel结果矩阵中与groudtruth相同的边
    out=[out;csave(i3,:) h_2(j3,:)]; %输出patel结果和标准网络相同的边组成的矩阵
end
    end
end
Num_fan=length(find(out(:,4)<=0)) ;
% 与groundtruth的对比方法，kappa是阈值，表示是否有边，tau表示方向，当tau为负值时，表示反向。
