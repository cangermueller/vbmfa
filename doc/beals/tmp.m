s = 1;  
p = 2;  % oberved
q = 2;  % hidden
qq = q+1;   %hidden+1

Lm{1} = [0 -0.013  0.0094;
         0  0.0943 -0.274];
     
Xxm{1} = ones(qq, 100);
Xxm{1}(2:qq, :) = ones(q, 100)*100;
Y = Lm{1}*Xxm{1};

psii = ones(p, 1);
Lcov{1} = zeros(qq, qq, p);
for i = 1:p,
    Lcov{1}(2:end, 2:end, i) = eye(q);
end