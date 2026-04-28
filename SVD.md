# SVD 怎么算：完整型与经济型

## 1. SVD 的基本形式

给定矩阵：

$$
A\in \mathbb{R}^{m\times n}
$$

SVD，即奇异值分解，可以写成：

$$
A=U\Sigma V^T
$$

其中：

$$
U\in \mathbb{R}^{m\times m},\ \Sigma\in \mathbb{R}^{m\times n},\ V\in \mathbb{R}^{n\times n}
$$

并且：

$$
U^TU=I_m,\qquad V^TV=I_n
$$

也就是说，在完整型 SVD 中，$U,V$ 都是正交矩阵。

---

## 2. SVD 的核心关系

SVD 的计算通常从两个矩阵入手：

$$
A^TA
$$

和

$$
AA^T
$$

其中：

$$
A^TA\in \mathbb{R}^{n\times n},\ AA^T\in \mathbb{R}^{m\times m}
$$

核心关系是：

$$
A^TA v_i=\sigma_i^2 v_i,\ AA^T u_i=\sigma_i^2 u_i
$$

所以：

- $v_i$ 是 $A^TA$ 的特征向量；
- $u_i$ 是 $AA^T$ 的特征向量；
- $\sigma_i$ 是奇异值；
- $\sigma_i^2$ 是 $A^TA$ 或 $AA^T$ 的非零特征值。

---

# 3. 完整型 SVD 的计算步骤

设：

$$
A\in \mathbb{R}^{m\times n}
$$

完整型 SVD 为：

$$
A=U\Sigma V^T
$$

其中：

$$
U\in \mathbb{R}^{m\times m},\ \Sigma\in \mathbb{R}^{m\times n},\ V\in \mathbb{R}^{n\times n}
$$

---

## 步骤 1：计算 $A^TA$

先计算：

$$
A^TA
$$

这是一个 $n\times n$ 的对称矩阵。

---

## 步骤 2：求 $A^TA$ 的特征值和特征向量

解：

$$
A^TA v_i=\lambda_i v_i
$$

把特征值从大到小排列：

$$
\lambda_1\geq \lambda_2\geq \cdots \geq \lambda_n\geq 0
$$

---

## 步骤 3：求奇异值

奇异值是特征值开平方：

$$
\sigma_i=\sqrt{\lambda_i}
$$

所以：

$$
\sigma_1\geq \sigma_2\geq \cdots \geq \sigma_n\geq 0
$$

---

## 步骤 4：构造 $V$

把 $A^TA$ 的单位特征向量按照奇异值从大到小排列：

$$
V=
\begin{bmatrix}
| & | & & |\\
v_1 & v_2 & \cdots & v_n\\
| & | & & |
\end{bmatrix}
$$

于是：

$$
V^TV=I_n
$$

---

## 步骤 5：计算左奇异向量 $u_i$

对于所有非零奇异值 $\sigma_i>0$，使用：

$$
u_i=\frac{Av_i}{\sigma_i}
$$

这样可以得到 $U$ 的一部分列向量。

如果还没有凑够 $m$ 个列向量，就继续补充正交单位向量，使得：

$$
U\in \mathbb{R}^{m\times m}
$$

并且：

$$
U^TU=I_m
$$

---

## 步骤 6：构造 $\Sigma$

$\Sigma$ 是一个 $m\times n$ 矩阵，主对角线上放奇异值：

$$
\Sigma=
\begin{bmatrix}
\sigma_1 & 0 & \cdots & 0\\
0 & \sigma_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots
\end{bmatrix}
$$

如果 $m\neq n$，多余的位置全部填 $0$。

---

# 4. 经济型 SVD 的计算步骤

设：

$$
r=\operatorname{rank}(A)
$$

经济型 SVD 只保留非零奇异值对应的部分：

$$
A=U_r\Sigma_rV_r^T
$$

其中：

$$
U_r\in \mathbb{R}^{m\times r},\ \Sigma_r\in \mathbb{R}^{r\times r},\ V_r\in \mathbb{R}^{n\times r}
$$

---

## 经济型 SVD 步骤

### 步骤 1：计算 $A^TA$

$$
A^TA
$$

---

### 步骤 2：求非零特征值

求出 $A^TA$ 的非零特征值：

$$
\lambda_1,\lambda_2,\dots,\lambda_r
$$

---

### 步骤 3：求非零奇异值

$$
\sigma_i=\sqrt{\lambda_i}
$$

---

### 步骤 4：构造 $V_r$

取这些非零特征值对应的单位特征向量：

$$
v_1,v_2,\dots,v_r
$$

组成：

$$
V_r=[v_1,v_2,\dots,v_r]
$$

---

### 步骤 5：构造 $U_r$

使用：

$$
u_i=\frac{Av_i}{\sigma_i}
$$

得到：

$$
U_r=[u_1,u_2,\dots,u_r]
$$

---

### 步骤 6：构造 $\Sigma_r$

$$
\Sigma_r=
\begin{bmatrix}
\sigma_1 & & 0\\
& \ddots & \\
0 & & \sigma_r
\end{bmatrix}
$$

于是：

$$
A=U_r\Sigma_rV_r^T
$$

---

# 5. 完整例子

取矩阵：

$$
A=
\begin{bmatrix}
1 & 1\\
1 & 1\\
0 & 0
\end{bmatrix}
$$

这是一个 $3\times 2$ 矩阵，因此：

$$
m=3,\qquad n=2
$$

---

## 5.1 计算 $A^TA$

先写出：

$$
A^T=
\begin{bmatrix}
1 & 1 & 0\\
1 & 1 & 0
\end{bmatrix}
$$

所以：

$$A^TA=\begin{bmatrix}
1 & 1 & 0\\
1 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 1\\
1 & 1\\
0 & 0
\end{bmatrix}
$$

得到：

$$
A^TA=
\begin{bmatrix}
2 & 2\\
2 & 2
\end{bmatrix}
$$

---

## 5.2 求 $A^TA$ 的特征值

解：

$$
\det(A^TA-\lambda I)=0
$$

即：

$$
\det
\begin{bmatrix}
2-\lambda & 2\\
2 & 2-\lambda
\end{bmatrix}
=0
$$

计算行列式，得到

$$
\lambda_1=4,\qquad \lambda_2=0
$$

---

## 5.3 求奇异值

$$
\sigma_i=\sqrt{\lambda_i}
$$

所以：

$$
\sigma_1=2,\qquad \sigma_2=0
$$

矩阵 $A$ 只有一个非零奇异值，因此：

$$
r=\operatorname{rank}(A)=1
$$

---

## 5.4 求右奇异向量 $v_i$

### 对 $\lambda_1=4$

解：

$$
(A^TA-4I)v=0
$$

即：

$$
\begin{bmatrix}
-2 & 2\\
2 & -2
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=0
$$

得到：

$$
x=y
$$

取单位向量：

$$
v_1=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

---

### 对 $\lambda_2=0$

解：

$$
A^TA v=0
$$

即：

$$
\begin{bmatrix}
2 & 2\\
2 & 2
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=0
$$

得到：

$$
x+y=0
$$

即：

$$
x=-y
$$

取单位向量：

$$
v_2=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
-1
\end{bmatrix}
$$

因此：

$$
V=\begin{bmatrix}
| & |\\
v_1 & v_2\\
| & |
\end{bmatrix}=\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

---

## 5.5 求左奇异向量 $u_1$

对于非零奇异值 $\sigma_1=2$：

$$
u_1=\frac{Av_1}{\sigma_1}
$$

先算：

$$
Av_1=\begin{bmatrix}
1 & 1\\
1 & 1\\
0 & 0
\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

得到：

$$Av_1=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
2\\
2\\
0
\end{bmatrix}=\begin{bmatrix}
\sqrt{2}\\
\sqrt{2}\\
0
\end{bmatrix}
$$

所以：

$$u_1=
\frac{1}{2}
\begin{bmatrix}
\sqrt{2}\\
\sqrt{2}\\
0
\end{bmatrix}=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}}\\
0
\end{bmatrix}
$$

---

# 6. 经济型 SVD 结果

因为：

$$
r=1
$$

所以经济型 SVD 只保留 $\sigma_1=2$。

于是：

$$
U_r=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}}\\
0
\end{bmatrix}
$$

$$
\Sigma_r=
\begin{bmatrix}
2
\end{bmatrix}
$$

$$
V_r=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

因此：

$$
A=U_r\Sigma_rV_r^T
$$

也就是：

$$
A=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}}\\
0
\end{bmatrix}
\begin{bmatrix}
2
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
\end{bmatrix}
$$

验证：

$$
U_r\Sigma_r=
\begin{bmatrix}
\sqrt{2}\\
\sqrt{2}\\
0
\end{bmatrix}
$$

所以：

$$U_r\Sigma_rV_r^T
=\begin{bmatrix}
\sqrt{2}\\
\sqrt{2}\\
0
\end{bmatrix}\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
\end{bmatrix}
=\begin{bmatrix}
1 & 1\\
1 & 1\\
0 & 0
\end{bmatrix}
$$

---

# 7. 完整型 SVD 结果

完整型 SVD 需要：

$$
U\in \mathbb{R}^{3\times 3}
$$

$$
\Sigma\in \mathbb{R}^{3\times 2}
$$

$$
V\in \mathbb{R}^{2\times 2}
$$

---

## 7.1 补齐 $U$

我们已经有：

$$
u_1=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}}\\
0
\end{bmatrix}
$$

还需要补两个正交单位向量，可以取：

$$
u_2=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
-\frac{1}{\sqrt{2}}\\
0
\end{bmatrix}
$$

$$
u_3=
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
$$

于是：

$$
U=
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0\\
0 & 0 & 1
\end{bmatrix}
$$

---

## 7.2 构造完整型 $\Sigma$

因为 $A$ 是 $3\times 2$，所以：

$$
\Sigma\in \mathbb{R}^{3\times 2}
$$

奇异值为：

$$
\sigma_1=2,\qquad \sigma_2=0
$$

因此：

$$
\Sigma=
\begin{bmatrix}
2 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
$$

---

## 7.3 写出 $V^T$

前面已经得到：

$$
V=
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

这个矩阵刚好是对称的，所以：

$$
V^T=
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

---

## 7.4 完整型 SVD

因此完整型 SVD 为：

$$
A=U\Sigma V^T
$$

即：

$$\begin{bmatrix}
1 & 1\\
1 & 1\\
0 & 0
\end{bmatrix}
=\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
2 & 0\\
0 & 0\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

---

# 8. 完整型与经济型对比

## 完整型 SVD

$$
A=U\Sigma V^T
$$

其中：

$$
U\in \mathbb{R}^{m\times m}
$$

$$
\Sigma\in \mathbb{R}^{m\times n}
$$

$$
V\in \mathbb{R}^{n\times n}
$$

特点：

- $U,V$ 都是完整正交矩阵；
- $\Sigma$ 与原矩阵 $A$ 同形状；
- 会补齐零空间方向；
- 写法完整，但有很多多余的零。

---

## 经济型 SVD

$$
A=U_r\Sigma_rV_r^T
$$

其中：

$$
U_r\in \mathbb{R}^{m\times r}
$$

$$
\Sigma_r\in \mathbb{R}^{r\times r}
$$

$$
V_r\in \mathbb{R}^{n\times r}
$$

特点：

- 只保留非零奇异值对应的部分；
- $U_r,V_r$ 通常不是方阵；
- 但它们列向量正交归一；
- 更简洁，更适合实际计算。

---

# 9. 手算 SVD 的通用模板

以后遇到矩阵 $A$，手算 SVD 按下面步骤：

## 第一步 计算 $A^TA$

## 第二步：求特征值 $A^TA v_i=\lambda_i v_i$

## 第三步：奇异值为 $\sigma_i=\sqrt{\lambda_i}$
## 第四步：求单位特征向量 $v_i$，组成：$V=[v_1,v_2,\dots,v_n]$

## 第五步：对非零奇异值，用：$u_i=\frac{Av_i}{\sigma_i}$ 得到左奇异向量。

## 第六步

构造经济型或完整型：

### 经济型

只保留非零奇异值：

$$
A=U_r\Sigma_rV_r^T
$$

### 完整型

补齐 $U,V$ 的正交基：

$$
A=U\Sigma V^T
$$

---

# 10. 最核心公式

$$
A^TA v_i=\sigma_i^2v_i
$$

$$
AA^T u_i=\sigma_i^2u_i
$$

$$
u_i=\frac{Av_i}{\sigma_i}
$$

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^T
$$

记住这几个式子，SVD 的计算就基本清楚了。