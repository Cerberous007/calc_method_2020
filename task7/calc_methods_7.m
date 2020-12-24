  
function tmp=myjacobi(k,n)
%функция возвращает массив полиномов Якоби p(i) в аналитическом виде,
%i=0,1,...,n,  i - степень полинома, k - верхний индекс.
pj=sym(zeros(n+1,1));
syms x;pj(1)=1;
switch n
    case 0
        tmp=pj;
    case 1
        pj(2)=(k+1)*x;
        tmp=pj;
    otherwise
        pj(2)=(k+1)*x;
        for i=2:n
            pj(i+1)=((i+k)/(i+2*k))*(2+(2*k-1)/i)*x*pj(i)-((i+k)/(i+2*k))*(1+(k-1)/i)*pj(i-1);
        end
        tmp=pj;
end

function [phi, dphi, ddphi]=myphi_dphi_d2kk_good_anal(k,n)
%функция возвращает массивы коодинатных функций phi(i), i=1,...,n
%в аналитическом виде и массивы их производных.
phi=sym(zeros(n,1));
dphi=sym(zeros(n,1));
ddphi=sym(zeros(n,1));
syms x;
jac=myjacobi(k,n-1);
%массив многочленов Якоби p(i), i=0,...,n-1
%в аналитическом виде,  k - верхний индекс.
djac=myjacobi(k-1,n);
for i=1:n
    phi(i)=(1-x^2)*jac(i);
    phi(i)=collect(phi(i));
    %приведение многочлена к виду, содержащему степени
    %с соответствующими коэффициентами.
    dphi(i)=-2*(i)*(1-x^2)^(k-1)*djac(i+1);
    dphi(i)=collect(dphi(i));
    ddphi(i)=-2*(i)*((k-1)*(1-x^2)^(k-2)*(-2*x)*djac(i+1)+(1-x^2)^(k-1)*(i+2*(k-1)+1)/2)*jac(i);
    ddphi(i)=collect(ddphi(i));
end
end


function [q_r, q_l] = bc(alpha,beta, a, b, p, y, z)
fun = p*y*z;
if or(alpha(1)==0,alpha(2)==0)
    q_l = 0;
else
    q_l = double(subs(fun,'x',a))*alpha(1)/alpha(2);
end
if or(beta(1)==0,beta(2)==0)
    q_r = 0;
else
    q_r = double(subs(fun,'x',b))*beta(1)/beta(2);
end
end

function [phi, dphi, ddphi] = coord_for_coloc(a,b,n,k)
syms 'x';
jac = myjacobi(k,n);
phi=sym(zeros(n,1));
dphi=sym(zeros(n,1));
ddphi=sym(zeros(n,1));
for i = 1:n
    phi(i) =(x-a)^2*(x-b)^2*jac(i);
    dphi(i) = diff(phi(i));
    ddphi(i) = diff(dphi(i));
end
end

function y = coloc(a,b,n, p, q, f,v)
ch = zeros(1,n);
for i=1:n
        ch(i)=(a+b)/2 + (a-b)/2*cos((2*i-1)*pi/(n*2));
end
k = 1;
%test2
% syms 'x';
% phi=sym(zeros(n,1));
% dphi=sym(zeros(n,1));
% ddphi=sym(zeros(n,1));
% phi(1) = 7/2*x^3 - 17/2*x + x^2;
% phi(2) = -1/7*x^2 -2/7*x + 1;
% dphi(1) = diff(phi(1));
% dphi(2) = diff(phi(2));
% ddphi(1) = diff(dphi(1));
% ddphi(2) = diff(dphi(2));
% [phi(3:n), dphi(3:n), ddphi(3:n)]=coord_for_coloc(a,b,n-2,k);

%test1
[phi, dphi, ddphi]=myphi_dphi_d2kk_good_anal(k,n);
d = zeros(n,1);
A = zeros(n);
for i = 1:n
    for j = 1:n
        func = -p*ddphi(j) - diff(p)*dphi(j)+v*dphi(j) + q*phi(j);
        A(i,j) = func(ch(i));
    end
    d(i) = f(ch(i));
end
digits(4);
alpha = vpa(A\d);
y = alpha.*phi;
y = sum(y);
end

function y = ritz(a,b,alpha, beta, n, p, q, f)
k = 1;
%test2
syms 'fun(x)';
fun(x) = x;
phi=sym(zeros(n,1));
dphi=sym(zeros(n,1));
phi(1) = 1;
phi(2) = fun;
dphi(1) = 0;
dphi(2) = diff(fun);
[phi(3:n), dphi(3:n), ~]=myphi_dphi_d2kk_good_anal(k,n-2);

%test1
%[phi, dphi, ~]=myphi_dphi_d2kk_good_anal(k,n);
d = zeros(n,1);
A = zeros(n);

for i = 1:n
    for j = 1:n
        [q_r, q_l] = bc(alpha,beta, a, b, p, phi(i), phi(j));
        A(i,j) = double(int(p*dphi(i)*dphi(j) + q*phi(i)*phi(j), a, b)) + q_r + q_l;
    end
    func = f*phi(i);
    d(i) = int(func, [a b]);
end
digits(4);
alpha = vpa(A\d);
y = alpha.*phi;
y = sum(y);
end



