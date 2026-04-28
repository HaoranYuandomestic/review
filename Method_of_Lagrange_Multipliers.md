# 拉格朗日乘子法
针对标准形式优化问题
$$\begin{array}{ll}
    \min\limits_{\boldsymbol{x}} & f(\boldsymbol{x})\\
    s.t. & g_i(\boldsymbol{x})\geqslant 0,\ i=1,\cdots,m
\end{array}$$
我们的求解步骤为
1) 构造拉格朗日函数
$$L(\boldsymbol{x},\boldsymbol{\alpha})=f(\boldsymbol{x})-\sum\limits_{i=1}^m\alpha_i g_i(\boldsymbol{x})$$
其中 $\alpha_i\geqslant 0$
2) 对 $L$ 中原始变量 $\boldsymbol{x}$ 求偏导，得到 $\boldsymbol{x},\boldsymbol{\alpha}$ 的关系式。
3) 带入：将关系式带入拉格朗日函数，消去 $\boldsymbol{x}$ ，得到只包含 $\boldsymbol{\alpha}$ 的函数，称此式为对偶函数。
4) 得到极大化问题：在约束条件下，最大化对偶函数。
5) 利用 KKT 条件求解问题：增加条件 $\alpha_i g_i(x)=0$ ，利用 KKT 求解规划问题。

# KKT 方法
这个就是我们在《运筹学》中所学习的方法，针对原始问题：
$$\begin{array}{ll}
    \min\limits_{\boldsymbol{x}} & f(\boldsymbol{x})\\
    s.t. & g_i(\boldsymbol{x})\geqslant 0,\ i=1,\cdots,m
\end{array}$$
他的 KKT 条件为
1) 平衡性：拉格朗日函数的梯度为零，即
$$\nabla_{\boldsymbol{x}}L(\boldsymbol{x}^*,\boldsymbol{\alpha}^*)=\nabla f(\boldsymbol{x}^*)-\sum\limits_{i=1}^m\alpha_i^*\nabla g_i(\boldsymbol{x}^*)=0$$
2) 原始可行性：我们应当保证原始给出的条件成立，即
$$g_i(\boldsymbol{x}^*)\geqslant 0,\ i=1,\cdots,m$$
3) 对偶可行性：我们要求拉格朗日乘子必须是非负数
$$\alpha^*_i\geqslant 0,\ i=1,\cdots,m$$
4) 互补松弛性：乘子和约束函数的乘积等于0
$$\alpha_i^* g_i(\boldsymbol{x^*})=0,\ i=1,\cdots,m$$