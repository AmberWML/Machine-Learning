%car resale
load car_data;

%car mpg
trn=300;
tst=size(data,1)-trn;
trn_dta=[ones(trn,1) data(1:trn,4)];
trn_val=mpg(1:trn,:);
tst_dta=[ones(tst,1) data(trn+1:size(data),4)];
tst_val=mpg(trn+1:size(data),:);
b = regress (trn_val, trn_dta);
trn_pred= b'*trn_dta';
tst_pred= b'*tst_dta';
%plot training data
plot(trn_dta(:,2),trn_val,'bo');
xlabel('Weight')
ylabel('mpg')
hold on
pause
%plot the linear regression line
x_grid = (1000:10:6000)';
x_grid_pred = b'* [ones(size(x_grid)) x_grid]';
plot(x_grid,x_grid_pred);
hold on
pause
plot(tst_dta(:,2),tst_pred,'rx');

% 
% %two dimensional case
 x1=data(:,4); %weight
 x2=data(:,2); %horsepower
 y=mpg;

%two dimensional case
figure;
X=[ones(size(x1)) x1 x2];
%b = regress(y,X) % Removes NaN data
b=(X'*X)^(-1)*X'*y
scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Weight')
ylabel('Horsepower')
zlabel('MPG')
view(50,10)

%polynomial fitting
figure;
load car_data;
X = [ones(size(x1)) x1 x2 x1.*x2 x1.*x1 x2.*x2];
b = regress(y,X)
%b=(X'*X)^(-1)*X'*y
scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT + b(5)*X1FIT.*X1FIT + b(6)*X2FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Weight')
ylabel('Horsepower')
zlabel('MPG')
view(50,10)


