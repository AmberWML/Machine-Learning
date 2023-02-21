load zoo;
t = fitctree(trn_data,trn_lab,'PredictorNames', fields);
view(t,'Mode','graph');
pause;
h = findall(groot,'Type','figure');
close(h);
for i=1:6
    imshow(strcat(tst_names{i},'.png'));
    set(gcf,'Position',[0 0 700 700]);
    title(tst_names{i},'FontSize',40);
    pause;
    y=predict(t,tst_data(i,:));
    title(sprintf('%s is %s',tst_names{i},y{1}),'FontSize',40,'color', 'r'); 
    pause;
end