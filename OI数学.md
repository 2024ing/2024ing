## 基础数学

### 快速幂

```cpp
#include<bits/stdc++.h>
using namespace std;
long long p;
long long Qpow(long long base,long long add){
	long long cnt=1;
	while(add){
		if(add&1)cnt=(cnt*base)%p;
		base=(base*base)%p;
		add>>=1;
	}
	return cnt;
}
int main(){
	long long a,b; 
	scanf("%lld%lld%lld",&a,&b,&p);
	printf("%lld^%lld mod %lld=%lld",a,b,p,Qpow(a,b));
	return 0;
}
```
### 质数
#### 判断质数
判断质数一般用试除法
最暴力的方法显然是从 $2$ 到 $n-1$ 逐个遍历检查
根据小学知识一个数的因数一般成对存在
所以只需遍历到 $\sqrt{x}$ 即可

```cpp
bool IsPrime(int key){
	for(int i=2;i<=key/i;i++)if(key%i==0)return false;
	return true;
}
```

#### 筛质数

##### 埃氏筛 
```cpp
void work(int lim){
	st[1]=true;
	for(int i=2;i<=lim;i++){
		if(st[i]==false){
			prime[++cnt]=i;
			for(int j=i+i;j<=n;j++)st[i]=true;
		}
	}
}
```

##### 线性筛
时间复杂度 $O(n)$
思路是只被其最小质因子筛去
我们用每一个数和所有质数相乘进行筛除
一开始能保证 $prime[j]$ 是乘积的最小质因子
直到 i 的最小质因子等于 $prime[j]$ 为止
此时直接 `break` 就好

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e8+9,M=1e7+9;
int prime[M],cnt;
bool st[N];
void Eular(int n){
	st[1]=true;
	for(int i=2;i<=n;i++){
		if(!st[i])prime[++cnt]=i;
		for(int j=1;j<=cnt&&i*prime[j]<=n;j++){
			st[i*prime[j]]=1;
			if(i%prime[j]==0)break;
		}
	}
}
int main(){
	int n,T,p;
	scanf("%d%d",&n,&T);
	Eular(n);
	while(T--){
		scanf("%d",&p);
		printf("%d\n",prime[p]);
	}
	return 0;
}
```



### 约数与余数

#### 前置知识：积性函数
如果一个函数 $Shit(x)$ 满足

$Shit(a) * Shit(b) = Shit(a*b)$

则称这个函数 $Shit(x)$ 为积性函数 

#### 欧拉函数

##### 基础

欧拉函数 $\phi(n)$ 指小于等于 $n$ 且与 $n$ 互质的数的个数 
欧拉函数性质:

1.  $\phi(1) =1$
因为 1 和 1 互质
2.  $\phi(prime)=prime-1$
从 $1 \to prime$ 中除 $prime$ 本身任何数都与 $prime$ 互质
3.  $\phi(prime^k)=prime^k-prime^{k-1}$
可知 $1 \to prime^k$ 中 只要不是 $prime$ 的倍数则都与原数互质
而在此范围内 $prime$ 的倍数有 $prime^k/prime=prime^{k-1}$ 个
所以用 $prime^k$ 减去该数即为结果
4.  $\phi$ 是积性函数

5.  若 $N= p[1]^{a[1]}*p[2]^{a[2]}* ... *  p[k]^{a[k]}$  
    则 $\phi(N)=N(1-1/a[1])(1-1/a[2])...(1-1/a[k])$
    由性质四 $\phi(N)=\phi(p[1]^{a[1]})*\phi(p[2]^{a[2]})* ... *\phi(p[k]*{a[k]})$
    然后把性质三带入
$$
\begin{aligned}
\phi(N)
&=(p[1]^{a[1]}-p[1]^{a[1]-1})*(p[2]^{a[2]}-p[2]^{a[2]-1})* ... *(p[k]^{a[k]}-p[k]^{a[k]-1})\\
&=p[1]^{a[1]}(1-1/a[1])* ... *p[k]^{a[k]}(1-1/a[k])\\
&=p[1]^{a[1]}p[2]^{a[2]} ... p[k]^{a[k]}*(1-1/a[1])*(1-1/a[2])* ...  *(1-1/a[k]) \\
&最终将 N 的唯一分解定理代入
phi(N)=N(1-1/a[1])*... *(1-1/a[k])
\end{aligned}
$$
##### 筛欧拉函数
我们发现这个东西欧拉筛可以干
当然根据欧拉筛的思路
我们主要使用已有的 phi 去得到其他数的 phi 值
分成三种情况

1. $k$ 是质数
   $phi(k)=k-1$

2. $GCD(k,prime) = 1$
   由积性函数性质
   $\phi(k*prime)=\phi(k)*\phi(prime)=(prime-1)*\phi(k)$

3. $GCD(k,prime) = prime$
   说明 $k$ 已有 $prime$ 这个因数
   看上面的唯一分解定理只有其中的 $N$ 会有所改变
   所以 $\phi(k*prime)=\phi(k)*prime$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e6+7;
int prime[N],phi[N],cnt;
bool st[N];
void Eular_Get_Phi(int Lim){
    st[1]=1,phi[1]=1;
    for(int i=2;i<=Lim;i++){
        if(st[i]==0){
            phi[i]=i-1;
            prime[++cnt]=i;
        }
        for(int j=1;j<=cnt&&(long long)prime[j]*i<=(long long)Lim;j++){
			st[i*prime[j]]=true;
            if(i%prime[j]==0){
                phi[i*prime[j]]=prime[j]*phi[i];
                break;
            }else{
                phi[i*prime[j]]=(prime[j]-1)*phi[i];
            }
        }
    }

}
int main(){
    int n;
    scanf("%d",&n);
    Eular_Get_Phi(n);
    long long ans=0;
    for(int i=1;i<=n;i++)ans+=phi[i];
    printf("%lld\n",ans);
    return 0;
}
```





#### 约数个数函数
##### 基础

用 $d(x)$ 表示
解析式如下:
若 $N= p[1]^{a[1]}*p[2]^{a[2]}* ... *  p[k]^{a[k]}$
则 $d(N)=(a[1]+1)*(a[2]+1)* ... *(a[k]+1)$
这个函数其实比较简单
显然每个质因数 $p[i]$ 都可以选择 $0\to k[i]$ 的次数
乘法原理即可得出原式
显然该函数为积性函数
(这就不需要玄学了,如果 $a,b$ 两个数互质则他们的乘积在唯一分解定理中不会有重叠,稍加思考即可得出)

##### 筛约数个数函数

1. $k$ 是质数
   $D(k)=2$

2. $GCD(k,prime) = 1$
   $D(k*prime)=D(k)*D(prime)=2*D(k)$

3. $GCD(k,prime) = prime$
   说明 $prime$ 是 $k$ 的最小质因子 $a[i]$
   显然 $D(k*prime)=(p[1]+1)*(p[2]+1)* ... *(p[i]+1+1)* ... *(p[k]+1)$
   所以我们还需要维护每个数最小质因子的次数 $MinRoot$
   1. $k$ 是质数
      $MinRoot(k)=1$
   
   2. $GCD(k,prime) = 1$
      $prime$ 不是 $k$ 的最小质因子
      $MinRoot(k*prime)=1$
   
   3. $GCD(k,prime) = prime$
      说明 $prime$ 是 $k$ 的最小质因子
      所以会将此次数 $+1$
      即 $MinRoot(k*prime)=MinRoot(k)+1$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1e6+7;
bool st[N];
int prime[N],cnt,MinRoot[N],D[N];
void Eular(int Lim){
    st[1]=1,D[1]=1;
    for(int i=2;i<=Lim;i++){
        if(!st[i]){
            prime[++cnt]=i;
            MinRoot[i]=1;
            D[i]=2;
   		}
        for(int j=1;j<=cnt&&(long long)i*prime[j]<=(long long)Lim;j++){
            st[i*prime[j]]=true;
            if(i%prime[j]==0){
                MinRoot[i*prime[j]]=MinRoot[i]+1;
                D[i*prime[j]]=D[i]/(MinRoot[i]+1)*(MinRoot[i]+2);
                break;
            }else{
                MinRoot[i*prime[j]]=1;
                D[i*prime[j]]=2*D[i];
            }
		}
	}
}
int main(){
    int n;
    scanf("%d",&n);
    Eular(n);
    long long ans=0;
    for(int i=1;i<=n;i++)ans+=D[i];
    printf("%lld\n",ans);
    return 0;
}    
```

##### 阶乘的约数个数

考虑使用解析式 $d(N)=(a[1]+1)*(a[2]+1)* ... *(a[k]+1)$
那我们就需要思考阶乘如何处理每个数的次数

```cpp
for(int i=1,tmp,p;i<=cnt;i++){
   tmp=0,p=prime[i];
   for(int j=p;j<=n;j*=p) tmp+=n/pow(p,j);
   sum*=tmp+1;
}
```

我们从每个数的不同次数考虑能够不重不漏地统计其次数

#### GCD

##### 辗转相除法

板子相信都会

```cpp
//dfs type
int GCD(int a,int b){return (b==0)?a:GCD(b,a%b);}

//loop type
int GCD(int a,int b){while(b!=0) swap(a,b),b=b%a; return a;}
```

首先我们设他们的 $\text{GCD}=k$
则 $a=xk,b=yk,GCD(x,y)=1$
我们首先设 $x>y$，若不满足跑一次循环就好了
那么每次我们所做的事情其实就是将 $x$ 减去若干个 $y$
由互质的性质得此时满足 $GCD(x',y)=1$
那么我们跳出循环的前一次
一定满足 $a$ 为 $b$ 的倍数，那么一定有 $y=1$，也就是满足 $b=k$

这个时间复杂度是稳定 $O(\log n)$ 的
只是处理高精度有点麻烦

##### 更相减损法

```cpp
int GCD(int a,int b,int count=1){
    if(b==0) return a*count;
    if(a<b) swap(a,b);
    while((a&1)==0&&(b&1)==0) count<<=1,a>>=1,b>>=1;
    return GCD(b,a-b,count);
}
```

这个基本高精度才会用
其实相当于辗转相除的弱化版，思路也很相似
理论上他是 $log$ 的
但比如这个数据就变成了 $O(n)$

```cpp
get 1000000 1
get 999999 1
get 999998 1
	......
get 1 1
return 1
```

所以有一个小优化
```cpp
int GCD(int a,int b,int count=1){
    if(b==0) return a*count;
    if(a<b) swap(a,b);
    while((a&1)==0&&(b&1)==0) count<<=1,a>>=1,b>>=1;
    while((a&1)==0) a>>=1;
    while((b&1)==0) b>>=1;
    return GCD(b,a-b,count);
}
```

#### exGCD

##### 贝祖定理

根据 $ax+by=GCD(a,b)$
$a,b$ 均为正整数
求 $x,y$ 的整数值。

------
因为 $ax+by=GCD(a,b)$
且 $GCD(a,b)=GCD(b,a\%b)$
所以 $ax+by=GCD(b,a\%b)$
由贝祖定理 $GCD(b,a\%b)=bX+a\%bY$
我们求出 $x,y$ 和下一层 $X,Y$ 的关系扔到递归里求解即可
等量代换 $ax+by=bX+a%bY$
又因为 $a%b=a-(a/b)*b$  (C++向下取整)
所以
$$
\begin{aligned}
ax+by&=bX+[a-(a/b)*b]Y\\
	 &=bX+aY-b*(a/b)*Y\\
	 &=aY+b[X-(a/b)*Y]\\
\end{aligned}
$$
所以得到
$$
\begin{cases}
x=Y\\
y=X-(a/b)*Y
\end{cases}
$$
但是递归需要出口
当 $b=0$ 时 $GCD(a,b)=a$
即当 $b=0$ 时，$ax+by=a$
把 $b$ 代入 $ax+0y=a$
显然此时 $x=1$
同时为了满足上一层 $y=0$

```cpp
int exGCD(int a,int b,int &x,int &y){
	if(b==0){
		x=1,y=0;
		return a;
	}
	int X,Y;
	int GCD_answer=exGCD(b,a%b,X,Y);
	x=Y;
	y=X-(a/b)*Y;
	return GCD_answer;
}
```

当然，该式子不是常见形式
我们可以直接把形参中的 $x,y$ 传进去

```cpp
int exGCD(int a,int b,int &x,int &y){
	if(b==0){
		x=1,y=0;
		return a;
	}
	int GCD_answer=exGCD(b,a%b,y,x);
	y-=(a/b)*x;
	return GCD_answer;
}
```

##### 线性同余方程
根据 $ax\%m=b$
$a,b,m$ 均为正整数
求 $x$

------
因为 $ax\%m=b$
其等价于 $ax=b+km$
所以 $ax-mk=b$
定义 $y=-k$
所以 $ax+my=b$
显然当 $b$ 为两数的 $GCD$ 倍数时有解

```cpp
#include<bits/stdc++.h>
using namespace std;
int exGCD(int a,int b,int &x,int &y){
	if(b==0){
		x=1,y=0;
		return a;
	}
	int GCD_answer=exGCD(b,a%b,y,x);
	y-=(a/b)*x;
	return GCD_answer;
}
int main(){
	int T,a,b,m,x,y,gcd_ans;
	scanf("%d",&T);
	while(T--){
		scanf("%d%d%d",&a,&b,&m);
		gcd_ans=exGCD(a,m,x,y);
		if(b%gcd_ans==0){
			printf("%d\n",(long long)x*b/gcd_ans%m);
		}else{
			puts("Shit!");
		}
	}
}
```

##### 逆元
根据 $ax\%m=1$ 求 $x$

------
同线性同余方程，要求 $a,m$ 互质

```cpp
#include<bits/stdc++.h>
#define int long long
using namespace std;
int exGCD(int a,int b,int &x,int &y){
	if(b==0){
		x=1,y=0;
		return a;
	}
	int GCD_answer=exGCD(b,a%b,y,x);
	y-=(a/b)*x;
	return GCD_answer;
}
signed main(){
	int T,a,m,x,y;
	scanf("%d",&T);
	while(T--){
		scanf("%d%d",&a,&m);
		if(exGCD(a,m,x,y)==1){
			printf("%d\n",(x%m+m)%m);
		}else{
			puts("impossible");
		}
	}
}
```

##### [其他操作](https://www.luogu.com.cn/problem/P5656)

这题是 exGCD 的全部操作
根本问题在于构造通解
设我们求出了一个解使得 $ax'+by'=c$
设 $gcd=gcd(a,b)$
易得
$$
\begin{cases}
x=x'+k*\frac b {gcd}\\
y=y'-k*\frac a {gcd}
\end{cases}
$$
有了这个求最小正整数解不难
如果其对应的另一个未知数的值为正
说明有正整数解
至于最大解则是另一个为最小值时所对应的解

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){x=1,y=0; return a;} int ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
int main(){
    ll a,b,c,x,y,gcd,tmpa,tmpb,xmax,ymax,xmin,ymin; int T;
    scanf("%d",&T);
    while(T--){
        scanf("%lld%lld%lld",&a,&b,&c),gcd=exgcd(a,b,x,y),x*=c/gcd,y*=c/gcd;
    	if(c%gcd!=0) puts("-1");
        else{
            tmpa=a/gcd,tmpb=b/gcd;
            xmin=(x%tmpb+tmpb)%tmpb,ymin=(y%tmpa+tmpa)%tmpa;
            if(xmin==0) xmin=tmpb; if(ymin==0) ymin=tmpa;
            xmax=(c-b*ymin)/a,ymax=(c-a*xmin)/b;
            if(xmax<=0) printf("%lld %lld\n",xmin,ymin);
            else printf("%lld %lld %lld %lld %lld\n",(xmax-xmin)/tmpb+1,xmin,ymin,xmax,ymax);
        }
    }
    return 0;
}
```

##### [扩展操作：CRT](https://www.luogu.com.cn/problem/P1495)

对于方程组
$$
\begin{cases}
n\%a_1=b_1\\
n\%a_2=b_2\\
\dots\\
n\%a_k=b_k
\end{cases}
$$
且满足其中的 $a$ 两两互质的情况下求 $n$ 的非负整数解

---
此算法其实就是一种构造
首先我们希望对于每一个式子往结果里搞一点"私活"却又不被其他模数发现
一个自然的想法是加入一个其他模数乘积的倍数
但发现这样是不会被发现了，~~然而不好弄私活了~~
于是想到再对于当前模数乘一个逆元
然后再把私活 $b$ 搞进去即可
最后发现原数加上或减去所有模数的乘积没有啥用
于是就来模它即为最小正整数解

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef __int128 ll;
const int N=15;
void exgcd(ll a,ll b,ll &x,ll &y){if(b==0) x=1,y=0; else exgcd(b,a%b,y,x),y-=a/b*x;}
ll inv(ll a,ll b){ll x,y; exgcd(a,b,x,y); return (x%b+b)%b;}
ll a[N],b[N],A=1,ans; int n;
template<class T> void read(T &x){
	char k=getchar(); x=0;
	while(!isdigit(k)) k=getchar();
	while(isdigit(k)) x=x*10+k-'0',k=getchar();
}
template<class T> void print(T x){
	stack<char> st;
	do st.push(x%10+'0'),x/=10; while(x);
	while(!st.empty()) putchar(st.top()),st.pop();
}
int main(){
    scanf("%d",&n); for(int i=1;i<=n;i++) read(a[i]),read(b[i]),A*=a[i];
    for(int i=1;i<=n;i++) ans=(ans+A/a[i]*inv(A/a[i],a[i])%A*b[i]%A)%A;
    print(ans);
    return 0;
}
```

##### [再进一步：exCRT](https://loj.ac/p/10214)

问题同上，不满足所有 $a$ 两两互质

---
将方程式两两合并，直至只剩一条即为答案
$$
\begin{cases}
n\%a_1=b_1\\
n\%a_2=b_2\\
\end{cases}
$$
我们将其简单转化一下


$$
\begin{aligned}
n&=k_1a_1+b_1=k_2a_2+b_2\\
k_1a_1-k_2a_2&=b_2-b_1\to \text{exGCD}\\
n&=(k_1+i*\frac{a_2}{gcd})a_1+b_1\\
&=k_1a_1+b_1+i*lcm\\
n\%lcm&=k_1a_1+b_1
\end{aligned}
$$
注意 $\text{exGCD}$ 可能无解所以需要特判

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef __int128 ll;
template<class T> inline void read(T &x){
    char k=getchar(); x=0;
    while(!isdigit(k)) k=getchar();
    while(isdigit(k)) x=x*10+k-'0',k=getchar();
}
template<class T> inline void print(T x){
    stack<char> st;
    do st.push('0'+x%10),x/=10; while(x);
    while(!st.empty()) putchar(st.top()),st.pop();
}
const int N=100009;
ll a[N],b[N];
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){x=1,y=0; return a;} ll ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
int main(){
    int n; ll tmp,x,y; scanf("%d",&n);
    for(int i=1;i<=n;i++) read(a[i]),read(b[i]);
    for(int i=n-1;i>=1;i--){
        tmp=exgcd(a[i],a[i+1],x,y),x*=(b[i+1]-b[i])/tmp;
        if((b[i+1]-b[i])%tmp!=0) printf("QwQ on case %d\n",i),exit(0); 
        b[i]+=x*a[i],a[i]=a[i]*a[i+1]/tmp,b[i]=(b[i]%a[i]+a[i])%a[i];
    }
    print(b[1]);
    return 0;
}
```

#### BSGS & exBSGS

##### [BGSG](https://www.luogu.com.cn/problem/P3846)

给出 $a,b,p$，满足 $p$ 为质数，对于 $a^k\equiv b\mod p$ 求 $x$ 的最小非负整数解 $(2\le b,n < p<2^{31})$

----

根据费马小定理 $a^{p-1}\equiv1\mod p$ 我们只需要枚举到 $p-1$ 即可
根据类似分块的方法我们将原始方程式写作 $a^{xA}\equiv ba^y\mod p$
然后我们 $0\to A-1$ 枚举 $y$ 并将 $ba^y\mod p$ 的值存入哈希表
接下来枚举 $x$ 并检查其是否在哈希表里即可
思想类似数论分块
那么我们时间复杂度即为 $O(A+\frac p A)$ 均值不等式 $A=\sqrt p\to O(\sqrt p)$

注意事项：

1. 如果 $a\%p=0$ 则可能在哈希表中存了 $0$，但由于不能除以 $0$ 
   当 $b$ 不为 $0$ 或 $1$ 时原方程无解
   需要进行特判
2. 若 $b=1$ 则答案为零
   原来的只能解决正整数解问题
3. 要适当多存入几个以防万一
4. 存哈希表的时候直接替换即可
   由于我们想要最小解
   所以我们尽量取次数较大的来除

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
unordered_map<ll,ll> mp;
ll Pow(ll a,ll x,ll p){
    ll ans=1,tmp=a%p;
    while(x){
        if(x&1) ans=ans*tmp%p;
        tmp=tmp*tmp%p,x>>=1;
    }
    return ans;
}
int main(){
    ll a,b,p,A,tmp;
    scanf("%lld%lld%lld",&p,&a,&b),a%=p,b%=p,A=sqrt(p)+1,tmp=Pow(a,A,p); 
    for(ll i=0,t=b;i<A;i++,t=t*a%p) mp[t]=i; 
    for(ll i=1,t=tmp;i<=A+1;i++,t=t*tmp%p) if(mp.count(t)) printf("%lld\n",i*A-mp[t]),exit(0);
    puts("no solution");
    return 0;
}
```



##### [exBGSG](https://www.luogu.com.cn/problem/P4195)

同上，不满足 $p$ 为质数

---

#### 逆元

逆元有两种求法
exGCD 的方法前面已经讲过，不再赘述
还有一种方法用了费马小定理
这个只能对于质数求逆元
就是 $a^{p-1}\equiv1\pmod p$
显然逆元可以快速幂求出

还有一种线性递推方式
我们求 $a$ 关于 $p$ 的逆元
设 $p=ka+b(b<a)$
即 $ka+b\equiv0\pmod p$
那么同乘 $inv_a$ 就等同于 $k+b*inv_a\equiv0\pmod p$
我们简单移一下项
$inv_a\equiv-k*inv_b\equiv\lfloor\frac ka\rfloor*inv_{k\bmod a}\pmod p$
这个就可以线性递推了

### 矩阵

#### 定义

矩阵就是一个二维数组 $A_{i,j}$ 表示其有 $i$ 行 $j$ 列，第 $i$ 行 $j$ 列的元素记作 $A(i,j)$
例如：
$$
A_{3,4}=
\begin{bmatrix}
3 & 6 & 8 & 4\\
1 & 1 & 9 & 1\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

特殊的，如果 $i=j$，这个矩阵又可称作 $i$ 阶方阵

#### 运算

1. 加法：把两个矩阵（要求行列个数相同）每个数进行相加
   $$
   \begin{bmatrix}
   3 & 6 & 8\\
   1 & 1 & 9\\
   \end{bmatrix}
   +
   \begin{bmatrix}
   7 & 6 & 4\\
   9 & 8 & 0
   \end{bmatrix}
   =
   \begin{bmatrix}
   3+7 & 6+6 & 8+4\\
   1+9 & 1+8 & 9+0\\
   \end{bmatrix}
   =
   \begin{bmatrix}
   10 & 12 & 12\\
   10 & 9 & 9
   \end{bmatrix}
   $$

2. 减法：直接相减，方法同上

3. 数乘：一个数乘一个矩阵，直接把数乘进每一个元素即可
   $$
   2*
   \begin{bmatrix}
   3 & 6 & 8\\
   1 & 1 & 9\\
   \end{bmatrix}
   =
   \begin{bmatrix}
   6 & 12 & 16\\
   2 & 2 & 18\\
   \end{bmatrix}
   $$

4. 矩阵乘法：两个矩阵相乘
   规定 $A_{i,j}*B_{j,k}=C_{i,k},C(m,n)=\sum^{k}_{i=1}{A(m,i)B(i,n)}$，矩乘不满足交换律（但满足结合律）
   要求必须前一个矩阵的列数和后一个矩阵的行数相同才可相乘
   口诀：**中间相同取两边**
   $$
   \\
   \begin{bmatrix}
   3 & 2 & 0\\
   1 & 1 & 3\\
   \end{bmatrix}
   *
   \begin{bmatrix}
   0 & 2\\
   1 & 0\\
   1 & 1
   \end{bmatrix}
   =
   \begin{bmatrix}
   3*0+2*1+0*1 & 3*2+2*0+0*1\\
   1*0+1*1+3*1 & 1*2+1*0+3*1
   \end{bmatrix}
   =
   \begin{bmatrix}
   2 & 6\\
   4 & 5
   \end{bmatrix}
   
   \\
   
   \begin{bmatrix}
   1\\
   2\\
   3
   \end{bmatrix}
   *
   \begin{bmatrix}
   1 & 2 & 3
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 2 & 3\\
   2 & 4 & 6\\
   3 & 6 & 9
   \end{bmatrix}
   $$
   
   ```cpp
   #include<bits/stdc++.h>
   using namespace std;
   const int N=109;
   stack<char> st;
   template<class T> class matrix{
   	public:
   		void scan(int _n,int _m){n=_n,m=_m; for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) k[i][j]=_scan();}
   		void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) _print(k[i][j]),putchar(' '); putchar('\n');} }
   		friend matrix operator *(matrix a,matrix b){
   			matrix c;
   			int n=a.n,m=b.n,p=b.m;
   			c.n=n,c.m=p;
   			for(int i=1;i<=n;i++) for(int j=1;j<=p;j++) for(int it=1;it<=m;it++) c.k[i][j]+=a.k[i][it]*b.k[it][j];
   			return c;
   		}
   		matrix(){}
   		matrix(int _n,int _m){n=_n,m=_m;}
   		matrix(int _n,int _m,T** x){
   			n=_n,m=_m;
   			for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) k[i][j]=x[i][j];
   		}
   		T k[N][N]={};
   		int n=0,m=0;
   	private:
   		T _scan(){
   			char k=getchar(); T x=0,op=1;
   			while(!isdigit(k)){if(k=='-') op*=-1; k=getchar();}
   			while(isdigit(k)) x=x*10+k-'0',k=getchar();
   			return x*op;
   		}
   		void _print(T x){
   			if(x<0) putchar('-'),x=-x;
   			do st.push(x%10+'0'),x/=10; while(x);
   			while(!st.empty()) putchar(st.top()),st.pop();
   		}
   };
   matrix<long long> A,B;
   int main(){
       int n,m,p;
       scanf("%d%d",&n,&m);
       A.scan(n,m);
       scanf("%d",&p);
       B.scan(m,p);
       (A*B).print();
       return 0;
   }
   ```
   
   如果所有矩阵和一个矩阵相乘都不变
   那我们称这个矩阵为单位矩阵
   不难发现这样的矩阵一定是一个方阵，无论左乘右乘
   下面以其他矩阵右乘单位矩阵为例研究其性质
   $$
   \begin{bmatrix}
   3 & 6 & 8 & 4\\
   1 & 1 & 9 & 1\\
   0 & 3 & 0 & 8
   \end{bmatrix}
   \begin{bmatrix}
   1 & 0 & 0 & 0\\
   0 & 1 & 0 & 0\\
   0 & 0 & 1 & 0\\
   0 & 0 & 0 & 1
   \end{bmatrix}
   =
   \begin{bmatrix}
   3 & 6 & 8 & 4\\
   1 & 1 & 9 & 1\\
   0 & 0 & 0 & 0
   \end{bmatrix}
   $$
   不难发现单位矩阵满足除了对角线上为一外其他位置均为零
   左乘同理
   
   矩乘看起来并没有什么用
   但是他可以帮助我们对一个递推式快速求值
   比如斐波那契数列 $a_i=a_{i-1}+a_{i-2}$
   我们发现每次只有两个数是有用的
   那我们就可以存成一个矩阵 $\begin{bmatrix}a_i\\a_{i+1}\end{bmatrix}$
   思考如何才能让它变成  $\begin{bmatrix}a_{i+1}\\a_i+a_{i+1}\end{bmatrix}$
   
   首先 $A_{2,1}$ 想变成 $C_{2,1}$
   既可以在右面乘一个 $B_{1,1}$（即所谓“右乘”）
   也可以在左面乘一个 $B_{2,2}$（即所谓”左乘“）
   显然应当对其左乘一个二二矩阵
   发现有性质 $(B(1,1)+B(1,2))a_i=a_{i+1},(B(2,1)+B(2,2))a_{i+1}=a_i+a_{i+1}$
   这貌似构造不出来
   我们再换个思路
   
   存矩阵 $\begin{bmatrix}a_i&a_{i+1}\end{bmatrix}\to\begin{bmatrix}a_{i+1}&a_i+a_{i+1}\end{bmatrix}$
   显然需要右乘一个二二矩阵
   则有性质$\begin{cases}B(1,1)a_i+B(2,1)a_{i+1}=a_{i+1}\\B(1,2)a_i+B(2,2)a_{i+1}=a_{i+1}+a_i\end{cases}$
   
   易得 $B=\begin{bmatrix}0 & 1\\1 & 1\end{bmatrix}$
   那我们想得到 $a_n$ 就可以直接 $\begin{bmatrix}1&1\end{bmatrix}\begin{bmatrix}0 & 1\\1 & 1\end{bmatrix}^{n-2}$ 得到 $\begin{bmatrix}a_{n-1} &a_n\end{bmatrix}$
   而后一项可以直接快速幂
   
   ```cpp
   #include<bits/stdc++.h>
   using namespace std;
   typedef long long ll;
   const int N=4;
   ll Mod;
   template<class T> struct matrix{
   		friend matrix operator *(matrix a,matrix b){
   			matrix c;
   			int n=a.n,m=b.n,p=b.m;
   			c.n=n,c.m=p;
   			for(int i=1;i<=n;i++) for(int j=1;j<=p;j++) for(int it=1;it<=m;it++) 
                   c.k[i][j]=(c.k[i][j]+a.k[i][it]*b.k[it][j])%Mod;
   			return c;
   		}
   		matrix(){n=m=0;}
   		matrix(int _n,int _m){n=_n,m=_m;}
   		matrix(int _n,int _m,T** x){
   			n=_n,m=_m;
   			for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) k[i][j]=x[i][j];
   		}
   		T k[N][N]={};
   		int n,m;
   };
   matrix<long long> A(1,2),B(2,2);
   template<class T> matrix<T> Pow(matrix<T> base,ll x){
       matrix<T> ans(2,2);
       ans.k[1][1]=ans.k[2][2]=1;
       while(x){
           if(x&1) ans=ans*base;
           x>>=1,base=base*base;
       }
       return ans;
   }
   int main(){
       ll n;
       scanf("%lld%lld",&n,&Mod);
       if(n==1||n==2) putchar('1'),exit(0);
       A.k[1][1]=A.k[1][2]=1;
       B.k[1][2]=B.k[2][1]=B.k[2][2]=1;
       printf("%lld\n",(A*Pow(B,n-2)).k[1][2]);
       return 0;
   }
   ```
   
#### 图上矩乘加速
##### [板子题](https://www.luogu.com.cn/problem/P2233)引入
我们用 dp 的思想
假设现在已经搞到原点到每个点走 $k$ 步的方案数
我们能否推出 $k+1$ 步的情况呢
发现这是可以的
我们对于每一个点 $dp_i$ 都可以从 $dp_{i-1}$ 和 $dp_{i+1}$ 转移而来（环的情况特判）
那显然有一个简单的递推思路

```cpp
#include<bits/stdc++.h>
using namespace std;
const int mod=1000,e=5;
int dp[10],tmp[10];
inline int f(int x){
    if(x==9) return 1;
    if(x==0) return 8;
    if(x==e) return 0;
    return x; 
}
inline void go(int x){tmp[x]=(dp[f(x-1)]+dp[f(x+1)])%mod;}
int main(){
    int n;
    scanf("%d",&n),dp[1]=1;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=8;j++) go(j);
        for(int j=1;j<=8;j++) dp[j]=tmp[j];
    }
    printf("%d\n",dp[e]);
}
```

然后就 A 了。。。
其实这个可以进行矩乘优化
我们将上一层答案记做一个矩阵 $A_{1,8}$
显然右乘一个八阶方阵即可
容易构造
$$
A=
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\\
B=
\begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 & 1\\
1 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
1 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
\end{bmatrix}
$$
但注意到 $dp_5$ 不能再去更新其他元素
那么第五行必定全零！

```cpp
#include<bits/stdc++.h>
using namespace std;
const int mod=1000,e=5,N=8;
inline int f(int x){x%=N; return (x==0)?N:x;}
template <class T> struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int j=1;j<=C.m;j++) for(int k=1;k<=A.m;k++) C.x[i][j]=(C.x[i][j]+A.x[i][k]*B.x[k][j])%mod;
        return C;
    }
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    int n,m;
    T x[N+2][N+2];
};
matrix<int> A(1,N),B(N,N);
template<class T> matrix<T> Pow(matrix <T> x,long long d){
    matrix<T> ans(x.n);
    while(d){
        if(d&1) ans=ans*x;
        x=x*x,d>>=1;
    }
    return ans;
}
int main(){
    int n; scanf("%d",&n),A.x[1][1]=1;
    for(int i=1;i<=N;i++) B.x[i][f(x+1)]=B.x[i][f(x-1)]=1;
    for(int i=1;i<=N;i++) B.x[e][i]=0;
    printf("%d\n",(A*Pow(B,n)).x[1][e]);
}
```

[P4159 迷路](https://www.luogu.com.cn/problem/P4159)

这道题点的数量和边权都非常小
思考拆点来做
然后和上一题就差不多了

```cpp
#include<bits/stdc++.h>
using namespace std;
const int mod=2009,N=999;
inline int f(int x){x%=N; return (x==0)?N:x;}
stack<char> st;
template <class T> struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int j=1;j<=C.m;j++) for(int k=1;k<=A.m;k++) C.x[i][j]=(C.x[i][j]+A.x[i][k]*B.x[k][j])%mod;
        return C;
    }
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    T x[N+2][N+2]={};
    void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) _print(x[i][j]),putchar(' '); putchar('\n');} }
    void _print(T f){do putchar(f%10+'0'),f/=10; while(f); while(!st.empty()) putchar(st.top()),st.pop();}
};
matrix<int> A,B;
template<class T> matrix<T> Pow(matrix <T> x,long long d){
    matrix<T> ans(x.n);
    while(d){
        if(d&1) ans=ans*x;
        x=x*x,d>>=1;
    }
    return ans;
}
char mp[15][15];
int n,cnt;
inline void Link(int l,int r,int len){
    if(len==0)return;
    int tmp=l;
    for(int i=1;i<len;i++) B.x[tmp][++cnt]=1,tmp=cnt;
    B.x[tmp][r]=1;
}
int main(){
	int t;
    scanf("%d%d",&n,&t),A.x[1][1]=1,cnt=n;
    for(int i=1;i<=n;i++) scanf("%s",mp[i]+1);
    for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) Link(i,j,mp[i][j]-'0');
    A.size(1,cnt),B.size(cnt,cnt);
	printf("%d\n",(A*Pow(B,t)).x[1][n]);
}
```

交一发，TLE 30pts
发现应该对于每一个点开虚点而非每一条边
![](https://img2024.cnblogs.com/blog/3590556/202506/3590556-20250602135803821-1844380721.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
const short mod=2009;
const int N=119;
inline int f(int x){x%=N; return (x==0)?N:x;}
stack<char> st;
template <class T> struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int j=1;j<=C.m;j++) for(int k=1;k<=A.m;k++) C.x[i][j]=(C.x[i][j]+A.x[i][k]*B.x[k][j])%mod;
        return C;
    }
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    T x[N+2][N+2]={};
};
matrix<short> A,B;
template<class T> matrix<T> Pow(matrix <T> x,int d){
    matrix<T> ans(x.n);
    while(d){
        if(d&1) ans=ans*x;
        x=x*x,d>>=1;
    }
    return ans;
}
char mp[15][15];
int n;
inline void Link(int l,int r,int len){
    if(len==0)return;
    int tmp=l;
    for(int i=1;i<len;i++) B.x[tmp][10*l+i]=1,tmp=10*l+i;
    B.x[tmp][r]=1;
}
int main(){
	int t;
    scanf("%d%d",&n,&t),A.x[1][1]=1;
    for(int i=1;i<=n;i++) scanf("%s",mp[i]+1);
    for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) Link(i,j,mp[i][j]-'0');
    A.size(1,110),B.size(110,110);
	printf("%d\n",(A*Pow(B,t)).x[1][n]);
}
```

##### [P2151 HH去散步](https://www.luogu.com.cn/problem/P2151)

这题初看板题
然而。。。

> 不保证任意两个路口之间至多只有一条路相连接
> 他不会立刻沿着刚刚走来的路走回
> 无向边

然而

> $N\le50,M\le60$

自然思考 $dp_{i,j}$ 为在第 $i$ 个节点刚刚走第 $j$ 条边的方案
然而我们肯定不能简单粗暴的直接进行 $3000$ 了
它跑 $n^3$ 必炸无疑
那我们能否对于每一条边给定专属于每个点的编号呢
貌似可以
我们通过复杂的构造后 ~~暂写作原来的形式搞式子~~
$dp_{i,j}=\sum_{x\to i\and id\neq j}dp_{x,all}$
这个式子大致不变
可以矩乘优化
~~TJ 里的点边互换是啥啊~~

```cpp
#include<bits/stdc++.h>
using namespace std;
const int mod=45989,N=130;
stack<char> st;
template <class T> struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int j=1;j<=C.m;j++) for(int k=1;k<=A.m;k++) C.x[i][j]=(C.x[i][j]+A.x[i][k]*B.x[k][j])%mod;
        return C;
    }
    friend matrix Pow(matrix x,long long d){
	    matrix ans(x.n);
	    while(d){
	        if(d&1) ans=ans*x;
	        x=x*x,d>>=1;
	    }
	    return ans;
	}
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    T x[N][N]={};
    void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) _print(x[i][j]),putchar(' '); putchar('\n');} }
    void _print(T f){do putchar(f%10+'0'),f/=10; while(f); while(!st.empty()) putchar(st.top()),st.pop();}
};
matrix<int> A,B;
int to[N],from[N],pos[N];
int main(){
	int n,m,t,l,r,cnt=0,ans=0;
	scanf("%d%d%d%d%d",&n,&m,&t,&l,&r);
	for(int i=1,ll,rr;i<=m;i++){
		scanf("%d%d",&ll,&rr);
		from[++cnt]=ll,to[cnt]=rr,pos[cnt]=i; 
		from[++cnt]=rr,to[cnt]=ll,pos[cnt]=i; 
	}
	from[++cnt]=n+1,to[cnt]=l,pos[cnt]=0;
	A.size(1,cnt),B.size(cnt,cnt),A.x[1][cnt]=1;
	for(int i=1;i<=cnt;i++) for(int j=1;j<=cnt;j++) if(pos[j]!=pos[i]&&to[j]==from[i]) B.x[j][i]=1;
	A=A*Pow(B,t);
	for(int i=1;i<=cnt;i++) if(to[i]==r) ans=(ans+A.x[1][i])%mod;
	printf("%d\n",ans);
}
```

##### [Codeforces 576D](https://codeforces.com/contest/576/problem/D)

- 给定一张 $n$ 个点 $m$ 条边的有向图，你要从 $1$ 号点走到 $n$ 号点。

- 已知当你走了 $d_i$ 条边之后才能走第 $i$ 条边。 

- 问最少要走多少条边才能到达 $n$ 号点。

- $n,m\le150,d_i\le10^9$

~~这题只能看 TJ 了~~
首先解锁咋搞？
我们发现在这样的题里只有一部分边是有用的
而动态的去考虑加入边很难搞
同时根据解锁的定义我们的边是按照解锁时间从小到大排列的
那么我们可以考虑按照 $d$ 对所有的边进行排序
然后用上一次维护的信息加一条边进行递推

这个是我们的根本思路
然后思考每加一条边后怎么判断当前是否有解
首先不是所有点都会作为起点
我们起码要保证所有的边都能走
那一定要求起点是至少走了 $d_i$ 条边的点

这一部分咋搞呢？
直接用邻接矩阵进行快速幂！（这回知道他为啥叫邻接矩阵了吧）
乘上一个初始矩阵 $[1\ 0\ 0\dots]$
然而注意这里他的唯一作用是维护可达性
所以我们只需要把他存成布尔类型
然后我们把加法换成或运算，乘法换成与运算

然后我们把他们作为起点进行多源 bfs（都塞队列里就行）
然后搞一个到终点的最短路
最后加上他们要走的就是答案

以上的这个方法一望而知是错的
它有可能只走限制小的边需要走很多很多
多到已经能够解除以后的限制
所以我们不能搜到一个答案之后就直接输出
而应当不断的加入所有边对于答案取最小值

~~CF 2700 不是给人做的~~

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=155;
struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int j=1;j<=C.m;j++) for(int k=1;k<=A.m;k++) C.x[i][j]|=(A.x[i][k]&B.x[k][j]);
        return C;
    }
    friend matrix Pow(matrix x,long long d){
	    matrix ans(x.n);
	    while(d){
	        if(d&1) ans=ans*x;
	        x=x*x,d>>=1;
	    }
	    return ans;
	}
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    bool x[N][N]={};
    void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) putchar(x[i][j]+'0'),putchar(' '); putchar('\n');} }
}basic,path,tmp;

struct line{
	int l,r,d;
	bool operator <(const line &rhs)const{
		return d<rhs.d;
	}
}o[N];

queue<int> q;
vector<int> c[N];
int dis[N],n,m,ans=INT_MAX;
void bfs(int t){
	tmp=basic*Pow(path,t);
	for(int i=1;i<=n;i++) if(tmp.x[1][i]) dis[i]=0,q.push(i); else dis[i]=N;
	while(!q.empty()){
		int top=q.front(); q.pop();
		for(int j:c[top]) if(dis[j]>dis[top]+1) dis[j]=dis[top]+1,q.push(j);
	}
	if(dis[n]!=N) ans=min(ans,dis[n]+t); 
}
int main(){
	scanf("%d%d",&n,&m),basic.size(1,n),path.size(n,n),basic.x[1][1]=1;
	for(int i=1;i<=m;i++) scanf("%d%d%d",&o[i].l,&o[i].r,&o[i].d); 
	sort(o+1,o+m+1);
	for(int i=1;i<=m;i++) c[o[i].l].push_back(o[i].r),bfs(o[i].d),path.x[o[i].l][o[i].r]=1; //注意顺序！ 
	if(ans==INT_MAX) puts("Impossible"); else printf("%d\n",ans);
	return 0;
}
```

喜提 TLE
这题邻接矩阵快速幂的部分可以 bitset 优化
我们再看一下当时的代码

```cpp
for(int i=1;i<=C.n;i++) 
    for(int j=1;j<=C.m;j++) 
        for(int k=1;k<=A.m;k++) 
            C.x[i][j]|=(A.x[i][k]&B.x[k][j]);
```

显然循环顺序可以更改

```cpp
for(int i=1;i<=C.n;i++) 
    for(int k=1;k<=A.m;k++)
        for(int j=1;j<=C.m;j++) 
            C.x[i][j]|=(A.x[i][k]&B.x[k][j]);
```

我们可以把矩阵开成 bitset 类型
然后注意到 ` A.x[i][k]` 不变于是直接

```cpp
for(int i=1;i<=C.n;i++) 
    for(int k=1;k<=A.m;k++)
         if(A.x[i][k])
            C.x[i]|=B.x[k][j];
```

改吧。。。
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=155;
struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int k=1;k<=A.m;k++) if(A.x[i][k]) C.x[i]|=B.x[k];
        return C;
    }
    friend matrix Pow(matrix x,long long d){
	    matrix ans(x.n);
	    while(d){
	        if(d&1) ans=ans*x;
	        x=x*x,d>>=1;
	    }
	    return ans;
	}
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    bitset<N> x[N];
    void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) putchar(x[i][j]+'0'),putchar(' '); putchar('\n');} }
}basic,path,tmp;

struct line{
	int l,r,d;
	bool operator <(const line &rhs)const{
		return d<rhs.d;
	}
}o[N];

queue<int> q;
vector<int> c[N];
int dis[N],n,m,ans=INT_MAX;
void bfs(int t){
	tmp=basic*Pow(path,t);
	for(int i=1;i<=n;i++) if(tmp.x[1][i]) dis[i]=0,q.push(i); else dis[i]=INT_MAX;
	while(!q.empty()){
		int top=q.front(); q.pop();
		for(int j:c[top]) if(dis[j]==INT_MAX) dis[j]=dis[top]+1,q.push(j);
	}
	if(dis[n]!=INT_MAX) ans=min(ans,dis[n]+t); 
}
int main(){
	scanf("%d%d",&n,&m),basic.size(1,n),path.size(n,n),basic.x[1][1]=1;
	for(int i=1;i<=m;i++) scanf("%d%d%d",&o[i].l,&o[i].r,&o[i].d); 
	sort(o+1,o+m+1);
	for(int i=1;i<=m;i++) c[o[i].l].push_back(o[i].r),bfs(o[i].d),path.x[o[i].l][o[i].r]=1; //注意顺序！ 
	if(ans==INT_MAX) puts("Impossible"); else printf("%d\n",ans);
	return 0;
}
```

一发入魂，WA on test 14 
发现有个小 bug
就是我们有可能在还没有解锁的情况下就将原来的边存到了图中默认可以直接走
正确的方法应是保留上一层的可达数据并乘上邻接矩阵的 $\Delta_d$ 次方即可

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=155;
struct matrix{
    friend matrix operator *(matrix A,matrix B){
        matrix C(A.n,B.m);
        for(int i=1;i<=C.n;i++) for(int k=1;k<=A.m;k++) if(A.x[i][k]) C.x[i]|=B.x[k];
        return C;
    }
    friend matrix Pow(matrix x,long long d){
	    matrix ans(x.n);
	    while(d){
	        if(d&1) ans=ans*x;
	        x=x*x,d>>=1;
	    }
	    return ans;
	}
    matrix(){n=m=0;}
    matrix(int _n){
        n=m=_n;
        for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) x[i][j]=0;
        for(int i=1;i<=n;i++) x[i][i]=1;
    }
    matrix(int _n,int _m){n=_n,m=_m;}
    inline void size(int _n,int _m){n=_n,m=_m;}
    int n,m;
    bitset<N> x[N];
    void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) putchar(x[i][j]+'0'),putchar(' '); putchar('\n');} }
}basic,path;
 
struct line{
	int l,r,d;
	bool operator <(const line &rhs)const{
		return d<rhs.d;
	}
}o[N];
 
queue<int> q;
vector<int> c[N];
int dis[N],n,m,ans=INT_MAX;
void bfs(int t,int delta){
	basic=basic*Pow(path,t);
	for(int i=1;i<=n;i++) if(basic.x[1][i]) dis[i]=0,q.push(i); else dis[i]=INT_MAX;
	while(!q.empty()){
		int top=q.front(); q.pop();
		for(int j:c[top]) if(dis[j]==INT_MAX) dis[j]=dis[top]+1,q.push(j);
	}
	if(dis[n]!=INT_MAX) ans=min(ans,dis[n]+delta); 
}
int main(){
	scanf("%d%d",&n,&m),basic.size(1,n),path.size(n,n),basic.x[1][1]=1;
	for(int i=1;i<=m;i++) scanf("%d%d%d",&o[i].l,&o[i].r,&o[i].d); 
	sort(o+1,o+m+1);
	for(int i=1;i<=m;i++){
		c[o[i].l].push_back(o[i].r);
		bfs(o[i].d-o[i-1].d,o[i].d);
		path.x[o[i].l][o[i].r]=1;
		 
	}	 
	if(ans==INT_MAX) puts("Impossible"); else printf("%d\n",ans);
	return 0;
}
```



### 组合数学

#### 基础知识

##### 排列组合定义

从 $n$ 个不同元素中，任取 $m$ 个元素排成一列考虑顺序的方案数称为排列，记作 $A_n^m$
若不考虑顺序则称作组合，记作 $C_n^m$
有基础公式 $A_n^m=\frac{n!}{(n-m)!},C_n^m=\frac{A_n^m}{m!}=\frac{n!}{m!(n-m)!}$

##### 插板法

我们思考几个问题：

1. 现有 $n$ 个 **完全相同** 的元素，要求将其分为 $k$ 组，保证每组至少有一个元素，一共有多少种分法？
   > 这个问题直接思考不好考虑，我们可以考虑将 $n$ 个元素变成 $n$ 个小球
   > 中间有 $n-1$ 个间隔
   > 然后往里面插入 $k-1$ 个隔板刚好能分成 $k$ 组
   > 于是答案即为 $C^{k-1}_{n-1}$

2. 如果每组可以为空呢？

   > 我们思考事先向原序列内插入 $k$ 个额外的球
   > 于是直接插板就会把他们分成 $k$ 组
   > 最后在每组拿走一个
   > 答案即为 $C_{n+k-1}^{k-1}$

3. 如果对于每一组都有下界限制 $a_i$ 呢？

   > 我们事先删掉 $\sum(a_i-1)=\sum{a_i}-k$ 个小球
   > 正常分组
   > 然后再根据需要放回每组 $a_i-1$ 个球
   > 答案即为 $C^{k-1}_{n+k-\sum a_i-1}$

#### 简单题目

##### [LibreOJ 2599. 计算系数](https://loj.ac/p/2599)

答案即为 $a^nb^mC_k^n$
```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll mod=10007;
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){x=1,y=0; return a;} ll tmp=exgcd(b,a%b,y,x); y-=a/b*x; return tmp;}
ll inv(ll a,ll b){ll x,y,tmp; tmp=b/exgcd(a,b,x,y); return (x%tmp+tmp)%tmp;}
int main(){
	ll a,b,k,n,m,ans=1;
	scanf("%lld%lld%lld%lld%lld",&a,&b,&k,&n,&m);
	for(int i=1;i<=n;i++) ans=ans*(k+1-i)%mod*inv(i,mod)%mod*a%mod;
	for(int i=1;i<=m;i++) ans=ans*b%mod;
	printf("%lld\n",ans);
	return 0;
}
```

##### [LibreOJ 10227. 2^k 进制数](https://loj.ac/p/10227)

首先这题要用高精度
然后我们可以枚举这个数有多少位
先考虑一种特殊情况：$w$ 是 $k$ 的倍数
假设共有 $l$ 位
这可以用插板法求解 $C^l_{2^k-1}$
显然我们枚举位数之后就是 $\sum_{l=2}^{w/k} C^l_{2^k-1}$
那如果他不整除呢？
我们直接枚举最高位
然后我们就可以仍然用插板法
$$
\sum_{i=1}^{limit} C^l_{2^k-1-i}
$$
加上原来的即为答案

```cpp
#include<bits/stdc++.h>
#define A a.k 
#define B b.k
using namespace std;
class num{
	public:
		num(){k.clear();}
		void clear(){k.clear();}
		void read(char *s){
			int len=strlen(s);
			k.clear();
			for(int i=len-1;i>=0;i--) k.push_back(s[i]-'0');
			while(!k.empty()&&k.back()==0) k.pop_back();
		}
		void print(){
			int len=k.size();
			if(len==0) putchar('0');
			for(int i=len-1;i>=0;i--) putchar(k[i]+'0');
		}
		friend num operator + (num a,num b){
			if(A.size()<B.size()) swap(a,b);
			int len=B.size();
			A.push_back(0);
			for(int i=0;i<len;i++) A[i+1]+=(A[i]+B[i])/10,A[i]=(A[i]+B[i])%10;
			for(int i=len;A[i]>=10;i++) A[i+1]++,A[i]%=10;
			if(A.back()==0) A.pop_back();
			return a;
		} 
		friend num operator * (num a,int b){
			int len=A.size();
			long long tmp=0;
			A.resize(len+15);
			for(int i=0;i<len||tmp!=0;i++){
				tmp+=b*A[i];
				A[i]=tmp%10;
				tmp/=10;
			}
			while(!A.empty()&&A.back()==0) A.pop_back();
			return a;
		} 
		friend num operator / (num a,int b){
			int len=A.size();
			long long tmp=0;
			for(int i=len-1;i>=0;i--){
				tmp=tmp*10+A[i];
				A[i]=tmp/b;
				tmp%=b;
			}
			while(!A.empty()&&A.back()==0) A.pop_back();
			return a;
		} 
	private:
		vector<char> k;
};
num ans;
num C(int m,int n){
	num tmp; tmp.read("1");
	if(m>n){tmp.read("0"); return tmp;	}
	for(int i=n-m+1;i<=n;i++) tmp=tmp*i;
	for(int i=1;i<=m;i++) tmp=tmp/i;
	return tmp;
}
int main(){
	int k,w,l,base,limit;
    scanf("%d%d",&k,&w),l=w/k,base=(1<<k)-1,limit=(1<<(w%k))-1;
    for(int i=2;i<=l;i++) ans=ans+C(i,base);//,printf("C(%d,%d)=",i,base),C(i,base).print(),putchar('\n'); 
    for(int i=1;i<=limit;i++) ans=ans+C(l,base-i);
    ans.print();
    return 0;
}
```

##### [LibreOJ 10228. 组合](https://loj.ac/p/10228)

这题显然直接乘会 TLE
对于组合数有一个递推式 $C_n^m=C_{n-1}^{m-1}+C_{n-1}^m$
我们其实就是在取 $n-1$ 个属的基础上加了一个数
要么不去取它就相当于没加
要么取了就相当于前面的名额少一个
但这就不用想了 $O(n^2)>1e18$ 必炸无疑
这个主要就是你要求一堆组合数的时候用

我们最简单的其实是直接算
直接按定义除逆元
但是当模数小于 $n$ 时会出问题
因为逆元不一定存在！

但这题不用管那么多
直接搞就行了

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll Pow(ll a,ll k,ll p){
    ll ans=1;
    while(k){
        if(k&1) ans=ans*a%p;
        k>>=1,a=a*a%p;
    }
    return ans;
}
int main(){
    int T; ll n,m,p,ans; scanf("%d",&T);
    while(T--){
        scanf("%lld%lld%lld",&n,&m,&p),ans=1;
        for(int i=1;i<=m;i++) ans=ans*Pow(i,p-2,p)%p*(n+1-i)%p;
        printf("%lld\n",ans);
    }
    return 0;
}

```

##### [LibreOJ 10230. 牡牛和牝牛](https://loj.ac/p/10230)

我们发现数据不用 Lucas
可以直接枚举公牛个数
然后我们假装给他搞好成 $\texttt{m fffm fffm fffm fffm... ("m" for male,"f" for female.)}$
接着找有几头牛没有插进去整一个插板法分组即可
所以答案为 $cnt_i=ik-k+1,\sum_{i=1}^{cnt_i<n} solve(n-cnt_i,i+1)=\sum_{i=1}^{cnt_i<n} C_{n-cnt+i}^i $
最后注意加上一种没有公牛的情况！

```cpp
#include<bits/stdc++.h>
using namespace std;
const int mod=5000011;
int n,k,inv[mod+5],fact[mod+5];
int C(int m,int n){return 1ll*fact[n]*inv[fact[m]]%mod*inv[fact[n-m]]%mod;}
int main(){
    int cnt=1,ans=1;
    scanf("%d%d",&n,&k),inv[1]=fact[0]=fact[1]=1;
    for(int i=2;i<mod;i++) fact[i]=1ll*fact[i-1]*i%mod,inv[i]=((-1ll*mod/i*inv[mod%i])%mod+mod)%mod;
   	for(int i=1;cnt<=n;i++,cnt+=k+1) ans=(ans+C(i,n-cnt+i))%mod;
    printf("%d",ans);
}
```

##### [LibreOJ 10231. 方程的解](https://loj.ac/p/10231)

这题好像是插板法板子题
$x^x\bmod 1000$ 就是个普通快速幂
方程就非空分组即可
那很简单了
不过需要高精度

```cpp
#include<bits/stdc++.h>
#define A a.k
using namespace std;
int k,x;
int Pow(int k,int n,int p){
	k%=p; int ans=1;
	while(n){
		if(n&1) ans=ans*k%p;
		n>>=1,k=k*k%p;
	}
	return ans;
}
class num{
	public:
		num(){k.clear(),k.push_back(1);}
		void print(){for(int i=k.size()-1;i>=0;i--) putchar(k[i]+'0');}
		friend num operator *(num a,int b){
			int len=A.size(),i;
			for(i=0;i<len;i++) A[i]*=b;
			A.resize(len+10);
			for(i=0;i<len||A[i]>9;i++) A[i+1]+=A[i]/10,A[i]%=10;
			while(A.size()>1&&A.back()==0) A.pop_back();
			return a;
		}
		friend num operator /(num a,int b){
			int len=A.size(); long long tmp=0;
			for(int i=len-1;i>=0;i--){
				tmp=tmp*10+A[i];
				A[i]=tmp/b;
				tmp%=b;
			}
			while(A.size()>1&&A.back()==0) A.pop_back();
			return a;
		}
	private:
		vector<int> k;
}ans;
int main(){
	scanf("%d%d",&k,&x),x=Pow(x,x,1000)-1,k--;
	if(x<k) putchar('0'),exit(0);
	for(int i=x;i>=x-k+1;i--) ans=ans*i;
	for(int i=1;i<=k;i++) ans=ans/i;
	ans.print();
	return 0;
}
```



#### Lucas

问题：求 $C^m_n\bmod p$ 的值 $(1\le n,m,p \le 1e6,p\ 为质数)$

引理一：
当满足 $x\neq0$ 且 $x\neq p$ 时，$C^x_p=\frac{p!}{x!(p-x)!}=\frac{p(p-1)!}{x(x-1)!(p-x)!}=\frac p xC_{p-1}^{x-1}\equiv p\operatorname{inv}(x)C_{p-1}^{x-1}\equiv0 \pmod p$
此时显然存在逆元，否则易得原式子的值为一
于是 $C_p^x=[x=0\ \text{or}\ x=p]$

引理二：
$(1+x)^k\equiv\sum_{i=0}^k C_k^ix^i\equiv1+x^k\pmod k$

引理三：
我们假设现在要求 $(1+x)^n\bmod p$ 的值（以下不区分等号和同余号）
$$
\begin{aligned}
(1+x)^n
&=
(1+x)^{p\lfloor \frac np\rfloor}(1+x)^{n\bmod p}
\\
&=
(1+x^p)^{\lfloor \frac np\rfloor}(1+x)^{n\bmod p}
\\
&=
\sum_{i=0}^{\lfloor \frac np\rfloor}
C_{\lfloor \frac np\rfloor}^i x^{ip}
\sum_{j=0}^{n\bmod p} 
C_{n\bmod p}^jx^j
\\
&=
\sum_{i=0}^{\lfloor \frac np\rfloor}
\sum_{j=0}^{n\bmod p} 
C_{\lfloor \frac np\rfloor}^i 
C_{n\bmod p}^j 
x^{ip+j}
\end{aligned}
$$
我们加入定义 $C_n^m(m>n)=0$
$$
\begin{aligned}
&=
\sum_{i=0}^{\lfloor \frac np\rfloor}
\sum_{j=0}^{p-1} 
C_{\lfloor \frac np\rfloor}^i 
C_{n\bmod p}^j 
x^{ip+j}
\\
&=
\sum_{i=1}^n
C_{\lfloor \frac np\rfloor}^{\lfloor \frac ip\rfloor}
C_{n\bmod p}^{i\bmod p} x^i
\end{aligned}
$$
而我们又知道
$$
(1+x)^n=\sum_{i=0}^n C_n^ix^i
$$
我们由于原式在模 $p$ 意义下为恒等式
所以可以猜测系数相同
所以有 Lucas 定理
$$
C_n^m\equiv C_{\lfloor\frac n p\rfloor}^{\lfloor\frac m p\rfloor}C^{m\%p}_{n\%p} \pmod p
$$

#### exLucas
问题同上，$1\le m\le n\le 10^{18},1\le q\le10^6,q=\prod p_i^{c_i},p\text{ is prime.}$

模仿《古代猪文》一题，将原模数拆分后在进行 CRT
那么我们的问题就是求 $C_n^m \bmod p^c=\frac{n!}{m!(n-m)!}\bmod p^c$
但一方面他会 T 另一方面逆元未必存在

逆元的问题当且仅当因子之中出现了 $p$ 的倍数
那可以思考把这些 $p$ 因子抽出来算
然后就有了一个抽象的式子
$$
\frac{n!}{m!(n-m)!}\bmod p^c=\frac{\frac{n!}{p^x}}{\frac{m!}{p^y}\frac{(n-m)!}{p^z}}p^{x-y-z}\bmod p^c 
$$
我们思考一下发现这个 $x-y-z>0$

> 因为我们这个数字中含 $p$ 的次数等于 $\sum \lfloor\frac{n}{p^i}\rfloor$
> 然后发现显然要把它合成一个较大的数字才有可能更大

所以我们这个部分用快速幂，剩下的就变成对于 $n$ 求 $\frac{n!}{p^x}\bmod p^c$

这个我们先举个例子
（注：$a\mid b$ 意思是 $a$ 是 $b$ 的因子）
$$
\begin{aligned}
n!
&=1*2*3*4*5*6*7*\dots*n\\
&=(p*2p*3p*\dots*\lfloor\frac{n}{p}\rfloor)(1*2*3*4*\dots)\\
&=p^{\lfloor\frac{n}{p}\rfloor}(\lfloor\frac{n}{p}\rfloor)!\prod_{i=1,p\not\mid i}^{n}i
\end{aligned}
$$
以下用 ”$=$“ 代表 "$\equiv\pmod {p^c}$"，并记 $P=p^c,\Delta=\lfloor\frac{n}{p}\rfloor,\Delta'=\lfloor\frac{n}{P}\rfloor$
$$
\begin{aligned}
&=p^{\Delta}\Delta!(\prod_{i=1,p\not\mid i}^{P\Delta'}i)(\prod_{i=P\Delta',p\not\mid i}^{n}i)\\
&=p^{\Delta}\Delta!(\prod_{i=1,p\not\mid i}^{P}i)^{\Delta'}(\prod_{i=P\Delta',p\not\mid i}^{n}i)\\
\end{aligned}
$$
定义 $\frac{n!}{p^x}\bmod p^c$ 为 $f(n)$
我们发现前面这个 $p^{\Delta}$ 一定要杀掉
但是 $\Delta!$ 也并不安全
那我们就得到了一个递归式 $f(n)=f(\Delta)(\prod_{i=1,p\not\mid i}^{P}i)^{\Delta'}(\prod_{i=P\Delta',p\not\mid i}^{n}i)$，显然边界 $f(0)=1$

下一个问题，怎么算 $p^{x-y-z}$？
说白了就是直接求 $n!$ 中的因子个数，设其为 $g(n)$
一方面，看上面他显然就是 $g(n)=\Delta g(\Delta)$
或者递推求

```cpp
ll ans=0; for(ll tmp=p;tmp<=n,tmp*=p) ans+=n/tmp;
```

我们在这里手推一组例子 $C_5^3 \bmod 12$
首先把 $12$ 拆成 $2^2*3$
对于 $2^2$，原式 $=\frac{5!}{2!*3!}=2*\frac{F(5)}{F(2)*F(3)}$
显然，$F(5)=15,F(2)=1,F(3)=3$
得到结果为 $10$，取模后为 $2$

对于 $3$，原式等于 $\frac{F(5)}{F(2)*F(3)}$ 
$F(5)=40,F(2)=2,F(3)=2$
取模完后为 $1$

可得同余方程组
$$
\begin{cases}
k\bmod4=2\\
k\bmod3=1
\end{cases}
$$
手算 $3$ 关于 $4$ 的逆元为 $3$，$4$ 关于 $3$ 的逆元为 $1$
$k=3*3*2+4=22\equiv10$

敲一下[P4720](https://www.luogu.com.cn/problem/P4720)

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef __int128 ll;
stack<char> show;
vector<ll> prime,cnt,ans;
template<class T> void scan(T &x){
    char k=getchar(); x=0;
    while(!isdigit(k)) k=getchar();
    while(isdigit(k)) x=x*10+k-'0',k=getchar();
}
template<class T> void print(T x){
    do show.push(x%10+'0'),x/=10; while(x);
    while(!show.empty()) putchar(show.top()),show.pop();
}
//ax+by=bX+(a-a/b*b)Y=aY-b(X-a/bY)
void exgcd(ll a,ll b,ll &x,ll &y){if(b==0) x=1,y=0; else exgcd(b,a%b,y,x),y-=a/b*x;}
ll inv(ll x,ll p){x%=p; ll m,n; exgcd(x,p,m,n); return (m%p+p)%p;}
ll Pow(ll n,ll m,ll p=1e18){
    n%=p; ll ans=1;
    while(m){
        if(m&1) ans=ans*n%p;
        n=n*n%p,m>>=1;
    }
    return ans;
}
ll F(ll n,ll p,ll P){
	if(n==0) return 1;
	ll delta=n/p,Delta=n/P,ans=1;
	for(ll i=1;i<=P;i++) if(i%p!=0) ans=ans*i%P;
	ans=Pow(ans,Delta,P);
	for(ll i=Delta*P;i<=n;i++) if(i%p!=0) ans=ans*i%P;
	return ans*F(delta,p,P)%P;
}
ll G(ll n,ll p){
	ll ans=0;
	for(ll tmp=p;tmp<=n;tmp*=p) ans+=n/tmp;
	return ans;
}
int main(){
    ll n,m,p,P,M,answer=0; scan(n),scan(m),scan(p),M=p;
    for(ll i=2;i*i<=p;i++){
        if(p%i!=0) continue;
        prime.push_back(i),cnt.push_back(0);
        while(p%i==0) p/=i,cnt[cnt.size()-1]++;
    }
    if(p>1) prime.push_back(p),cnt.push_back(1);
    for(int i=0;i<cnt.size();i++){
//    	printf("test "),print(prime[i]),putchar('^'),print(cnt[i]),putchar('\n');
    	p=prime[i],P=Pow(prime[i],cnt[i]);
    	ans.push_back(F(n,p,P)*inv(F(m,p,P),P)%P*inv(F(n-m,p,P),P)%P*Pow(p,G(n,p)-G(m,p)-G(n-m,p),P)%P);
//    	print(ans[ans.size()-1]),putchar('\n');
		prime[i]=Pow(prime[i],cnt[i]);
	}
	for(int i=0;i<ans.size();i++){
		p=prime[i];
		answer=(answer+M/p*inv(M/p,p)%M*ans[i]%M)%M;
	}
	print(answer);
	return 0;
}
```

#### Catalan

> Catalan 数列可以应用于以下问题：
>
> 1. 有 $2n$ 个人排成一行进入剧场。入场费 $5$ 元。其中只有 $n$ 个人有一张 $5$ 元钞票，另外 $n$ 人只有 $10$ 元钞票，剧院无其它钞票，问有多少种方法使得只要有 $10$ 元的人买票，售票处就有 $5$ 元的钞票找零？
> 2. 有一个大小为 $n\times n$ 的方格图左下角为 $(0, 0)$ 右上角为$ (n, n)$，从左下角开始每次都只能向右或者向上走一单位，不走到对角线 $y=x$ 上方（但可以触碰)的情况下到达右上角有多少可能的路径？
> 3. 在圆上选择 $2n$ 个点，将这些点成对连接起来使得所得到的 $n$ 条线段不相交的方法数？
> 4. 对角线不相交的情况下，将一个凸多边形区域分成三角形区域的方法数？
> 5. 一个栈（无穷大）的进栈序列为 $1,2,3, \cdots ,n$有多少个不同的出栈序列？
> 6. $n$ 个结点可构造多少个不同的二叉树？
> 7. 由 $n$ 个 $+1$ 和 $n$ 个 $-1$ 组成的 $2n$ 个数 $a_1,a_2, \ cdots , a _ { 2n } $，其部分和满足 $a_1+a_2+ \cdots +a_k \geq 0~(k=1,2,3, \cdots ,2n)$，有多少个满足条件的数列？
>
> ——OI-Wiki

简而言之，对于两个参数 $a$ 和 $b$，每次操作可以任选一个加一，但需时刻满足 $a\le b$，求获得 $(n,n)$ 的方案数
显然，他和其中的一二五七是本质相同的
第三个考虑每一个点顺时针排列有"入点"和"出点"两种选择（开始挂线/接上原来离得最近的一条挂好的）

这个数咋算呢？
代码是容易的

```cpp
int dp[N][N];
void get_catalan(int n){
    dp[0][0]=1;
    for(int i=0;i<=n;i++) for(int j=0;j<=i;j++){
        if(j>0) dp[i][j]+=dp[i][j-1];
        if(i>j) dp[i][j]+=dp[i-1][j];
    }
}
```

首先有一个结论，对于普通的卡特兰数，解析式为 

$Catalan(n)
=C_{2n}^n-C_{2n}^{n-1}\\
=\frac{(2n)!}{(n!)^2}-\frac{(2n)!}{(n-1)!(n+1)!}\\
=\frac{(2n)![\frac{n!n!(n+1)}{n}-n!n!]}{(n-1)!n!n!(n+1)!}\\
=\frac{(2n)!n!n!}{n(n-1)!n!n!(n+1)!}\\
=\frac{(2n)!}{n(n-1)!(n+1)!}\\
=\frac{(2n)!}{n!(n+1)!}\\
=\frac{C_{2n}^n}{n+1}
$
递推式为 $Catalan(n)=\sum_{i=0}^{n-1}(Catalan(i)*Catalan(n-i-1))$
更标准的递推
$Catalan(n)=\frac{(2n)!}{n!(n+1)!}\\
=\frac{(2n-2)!(2n-1)2n}{n!(n-1)!n(n+1)}\\
=\frac{4n+2}{n+1}Catalan(n-1)$

> 推导：[「算法入门笔记」卡特兰数 - 知乎](https://zhuanlan.zhihu.com/p/97619085)

二叉树的用 dp/dfs 随便搞一下就有
凸多边形呢？显然 $k$ 边形能有 $k-2$ 条边
$f(n)=\sum_{i=1}^{n-1}f(i)f(n-1-i)$ 同 $Catalan$ 
显然，对于 $n$ 边形划分为 $Catalan(n-2)$

> 附：Catalan 数前几项
> $1,1,2,5,14,42,132,429,1430,4862,16796,58786,208012,742900,2674440,9694845,35357670,129644790$

#### exCatalan

虽然但是。。。并没有这个名字。。。
显然，我们会遇到很多 $(0,0)\to(m,n)\quad m\ge n$ 的情况
还是用 $\text{Leetcode}$ 的推导方式
显然通项
$C_{m+n}^m-C_{m+n}^{m+1}\\
=\frac{(m+n)![m!n!(m+1)/n-m!n!]}{m!n!(m+1)!(n-1)!}\\
=\frac{(m+n)![(m+1)/n-1]}{(m+1)!(n-1)!}\\
=\frac{(m+n)![(m+1)/n-1]}{(m+1)!(n-1)!}\\
=\frac{(m+n)!(m-n+1)}{(m+1)!n!}\\$

### 博弈论

~~ybt 终于作了回人~~
第一题[取石子游戏](http://www.accoders.com/problem.php?cid=2305&pid=0) 
博弈论中有一些基本概念

> 公平组合游戏的定义如下：
>
> - 游戏有两个人参与，二者轮流做出决策，双方均知道游戏的完整信息；
> - 任意一个游戏者在某一确定状态可以作出的决策集合只与当前的状态有关，而与游戏者无关；
> - 游戏中的同一个状态不可能多次抵达，游戏以玩家无法行动为结束，且游戏一定会在有限步后以非平局结束。

> 定义 **必胜状态** 为 **当前操作者必胜的状态**，**必败状态** 为 **当前操作者必败的状态**。

显然，按照 $dp$ 的思路状态很好设

> - 定理·0：一个状态不是必胜状态就是必败状态。
> - 定理 1：没有后继状态的状态是必败状态。（$dp_0$）
> - 定理 2：一个状态是必胜状态当且仅当存在至少一个必败状态为它的后继状态。（$dp_{1,2,3,\dots,k}$）
> - 定理 3：一个状态是必败状态当且仅当它的所有后继状态均为必胜状态。（$dp_{k+1}$）

其实玩过这个游戏的都知道
我们只要给对方剩下一个 $dp_{\Delta(k+1)},\Delta\in\N_+$ 即可
我们每次一定能配齐 $k+1$ 个数
最后一定能取得最后一个
但如果你上来就一个必败肯定不行
显然

> $$
> \begin{cases}
> k+1\mid n\quad lose\\
> k+1\not\mid n\quad win\\
> \end{cases}
> $$

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    int n,k; scanf("%d%d",&n,&k); printf("%d",(n%(k+1)==0)?2:1);
    return 0;
}
```

---

第二题[取石子游戏 2](https://loj.ac/p/10242)
这个又叫 Nim 游戏
我们首先想：最简单的必败状态显然是全零
从而，只有一个非零是必胜状态
两个非零呢？
显然双一必败
扩展一下就能得到偶一必胜甚至全对必败

问题并没有改善

> SG 函数：
> 定义必败态 $g(fail)=0$
> 否则 $g(current)=\operatorname{mex}\{g(next)\}$，其中 $\operatorname{mex}$ 代表集合中未出现的最小的整数

> 性质：（小写代表元素，大写代表可重集）
>
> 1. $g(x)=x$
> 2. $g(X,Y)=g(Y,X)$
> 3. $g(X,X)=0$
> 4. $g(X)=k,g(Y)=0\to g(X,Y)=k$

> SG 定理：
> $g(x_1,x_2,\dots)=\bigoplus_{i=1}^n g(x_i)$

或者。。。[**Link**](https://zhuanlan.zhihu.com/p/358979118)

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    int T,n,k,tmp;
    scanf("%d",&T);
    while(scanf("%d",&n)!=EOF){
        tmp=0;
        while(n--) scanf("%d",&k),tmp^=k; 
        printf("%s\n",(tmp==0)?"No":"Yes");
    }
    return 0;
}
```

---

第三题[移棋子游戏](http://www.accoders.com/problem.php?cid=2305&pid=2)
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=2034;
vector<int> k[N],r[N];
int g[N],in[N];
queue<int> q;
unordered_set<int> mp;
int main(){
	int n,m,t,top,ll,rr; scanf("%d%d%d",&n,&m,&t);
    while(m--) scanf("%d%d",&ll,&rr),r[ll].push_back(rr),k[rr].push_back(ll),++in[ll];
    for(int i=1;i<=n;i++) if(in[i]==0) q.push(i);
    while(!q.empty()){
        top=q.front(),q.pop(),mp.clear();
        for(int i:r[top]) mp.insert(g[i]);
        for(int i=0;i<=n+1;i++) if(!mp.count(i)){g[top]=i;  break;}
       	for(int i:k[top]) if(--in[i]==0) q.push(i);
    }
    rr=0;
    while(t--) scanf("%d",&ll),rr^=g[ll];
    printf("%s",(rr!=0)?"win":"lose");
	return 0;
}
//g++ -std=c++14 -O2 -Wall tmp.cpp -o tmp; .\tmp

```

---

第四题[取石子游戏](http://www.accoders.com/problem.php?cid=2305&pid=3)
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1034,M=10034;
vector<int> k[M],r[M];
int g[M],in[M],num[N],lim[N],cnt,tmp[N];
queue<int> q;
unordered_set<int> mp;
void Link(int ll,int rr){
	// printf("Link %d %d\n",ll,rr);
	k[rr].push_back(ll),++in[ll];
	r[ll].push_back(rr);
}
int main(){
	int T,m,top,x=0; scanf("%d",&T);
	for(int i=1;i<=T;i++) scanf("%d",&num[i]);
	scanf("%d",&m); for(int i=1;i<=m;i++) scanf("%d",&lim[i]); sort(lim+1,lim+m+1);
	for(int i=1;i<=T;i++){
		++cnt;
		for(int j=1;j<=num[i];j++) for(int o=1;lim[o]<=j&&o<=m;o++) Link(j+cnt,j+cnt-lim[o]);
		cnt+=num[i],tmp[i]=cnt;
	}
	for(int i=1;i<=cnt;i++) if(in[i]==0) q.push(i);
    while(!q.empty()){
        top=q.front(),q.pop(),mp.clear();
        for(int i:r[top]) mp.insert(g[i]);
        for(int i=0;i<=cnt+1;i++) if(!mp.count(i)){g[top]=i; /*printf("g[%d]=%d\n",top,g[top]);*/ break;}
       	for(int i:k[top]) if(--in[i]==0) q.push(i);//,printf("push %d\n",i);
    }
	for(int i=1;i<=T;i++) x^=g[tmp[i]];
	if(x==0) puts("NO"),exit(0);
	puts("YES");
	for(int i=1;i<=T;i++) for(int j=1;lim[j]<=num[i]&&j<=m;j++) if((x^g[tmp[i]]^g[tmp[i]-lim[j]])==0) printf("%d %d\n",i,lim[j]),exit(0);
	return 0;
}
```

---

第五题[巧克力棒](http://www.accoders.com/problem.php?cid=2305&pid=4)
数据来看 $n$ 极小 $m$ 极大
只需取出一个异或和为零的必败给对方
对方无论吃还是取都可调为必败
但注意我们应当尽量多去取
比如你搞了一个全零
对方又拿了一个全零
那你不成 joker 了吗？

但他没要方案
乱搞即可

```cpp
#include<bits/stdc++.h>
using namespace std;
int n,k[20];
int main(){
	while(scanf("%d",&n)!=EOF){
		for(int i=0;i<n;i++) scanf("%d",&k[i]);
		for(int i=1,tmp;i<(1<<n);i++){
			tmp=0;
			for(int j=0;j<n;j++) if(i&(1<<j)) tmp^=k[j];
			if(tmp==0){puts("NO");goto here;}
		}
		puts("YES");
		here:;
	}
	return 0;
}
// g++ -std=c++14 -O2 -Wall tmp.cpp -o tmp; .\tmp
```

---

第六题[取石子](https://loj.ac/p/10246)
这个题主要是如果取没了就不能合并了
注意是合并任意两堆
发现对于只有一个的堆比较特殊，记作 $\text{single}$
其余的合并对答案无影响，略去
$\texttt{define dp[single][cnt],cnt represents normal heap takens.}$
显然有转移
$\texttt{dp[single][cnt]=dp[single-1][cnt],dp[single][cnt-1],dp[single-2][cnt+2/3],dp[single-1][cnt+1]}$
上四项分别代表：取 $\texttt{single}$，合并（取）两正常，合并两 $\texttt{single}$，合并一正常一 $\texttt{single}$
现在有个问题：有没有可能把一个正常堆删成了 $\texttt{single}$ 呢
考虑如果你需要，对手反手一合并，就没有意义了
若不需要，对手反手删了，你就寄了
然而，当 $cnt=1$ 时需要考虑

```cpp 
#include<bits/stdc++.h>
#define debug(x) {/*printf("dp(%d,%d)=1 from case %d\n",n,m,x);*/ Dp[n][m]=1; return 1;}
using namespace std;
char Dp[51][55000];
bool dp(int n,int m){
	// printf("goto %d %d\n",n,m);
	if(n==0&&m==0) return 0;
	if(m==1) ++n,--m;
	if(Dp[n][m]!=-1) return Dp[n][m];
	// printf("not in set\n");
	if(n>1) if(dp(n-2,m+(m?3:2))==0) debug(1);
	if(n>0&&m>0) if(dp(n-1,m+1)==0) debug(2);
	if(m>0) if(dp(n,m-1)==0) debug(3);
	if(n>0) if(dp(n-1,m)==0) debug(4);
	// printf("dp(%d,%d)=0\n",n,m);
	Dp[n][m]=0; return 0;
}
int main(){
	int T,n,a[1024],cnt,tot; scanf("%d",&T);
	for(int i=0;i<=50;i++) for(int j=0;j<=50000;j++) Dp[i][j]=-1;
	while(T--){
		scanf("%d",&n),cnt=0,tot=0; for(int i=1;i<=n;i++) scanf("%d",&a[i]),cnt+=(a[i]==1),tot+=(a[i]!=1)?a[i]+1:0;
		puts((dp(cnt,max(0,tot-1)))?"YES":"NO");
	}
	return 0;
}
// g++ -std=c++14 -O2 -Wall tmp.cpp -o tmp; .\tmp
/*
Hack 01:
1
5
1 2 1 2 2 

Output 01:

NO
*/
```

---

第七题[S-Nim](http://www.accoders.com/problem.php?cid=2305&pid=6)
以为把 T4 改改就行
其实不然
我们发现原来思路过于屎了
直接搞一次就行啊！
TLE code

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1034,M=1000034;
vector<int> k[M],r[M];
int g[M],in[M],num[N],lim[N],cnt,tmp[N];
queue<int> q;
unordered_set<int> mp;
void Link(int ll,int rr){
	// printf("Link %d %d\n",ll,rr);
	k[rr].push_back(ll),++in[ll];
	r[ll].push_back(rr);
}
int main(){
	freopen("test.in","r",stdin);
	freopen("output.txt","w",stdout);
	int T,T2,m,top,x=0; 
	int start=clock();
	while(1){
		scanf("%d",&m); if(m==0) return 0; for(int i=1;i<=m;i++) scanf("%d",&lim[i]); sort(lim+1,lim+m+1);
		scanf("%d",&T2); 
		while(T2--){
			scanf("%d",&T); cnt=0; for(int i=1;i<=T;i++) scanf("%d",&num[i]); 
			for(int i=1;i<=T;i++){
				++cnt;
				for(int j=1;j<=num[i];j++) for(int o=1;lim[o]<=j&&o<=m;o++) Link(j+cnt,j+cnt-lim[o]);
				cnt+=num[i],tmp[i]=cnt;
			}
			for(int i=1;i<=cnt;i++) if(in[i]==0) q.push(i);
			while(!q.empty()){
				top=q.front(),q.pop(),mp.clear();
				for(int i:r[top]) mp.insert(g[i]);
				for(int i=0;i<=cnt+1;i++) if(!mp.count(i)){g[top]=i; break;}
				for(int i:k[top]) if(--in[i]==0) q.push(i);
			}
			for(int i=1;i<=T;i++) x^=g[tmp[i]];
			if(x==0) putchar('L'); else putchar('W');
			for(int i=1;i<=cnt;i++) k[i].clear(),r[i].clear(),g[i]=0,in[i]=0; cnt=0,x=0;
		}		
		putchar('\n');
		printf("%d ms used\n",clock()-start);
		if(clock()-start>15000) puts("TLE"),exit(0);
	}
	return 0;
}

/*
2 2 5
3
2 5 12
3 2 4 7
4 2 3 7 12
5 1 2 3 4 5
3
2 5 12
3 2 4 7
4 2 3 7 12
0
*/
```

AC code（上火车头）

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1034,M=10034,cnt=10000;
vector<int> k[M],r[M];
int g[M],in[M],num[N],lim[N];
queue<int> q;
unordered_set<int> mp;
void Link(int ll,int rr){
	k[rr].push_back(ll),++in[ll];
	r[ll].push_back(rr);
}
int main(){
	// freopen("test.in","r",stdin);
	// freopen("output.txt","w",stdout);
	int T,T2,m,top,x=0; 
	// int start=clock();
	while(1){
		scanf("%d",&m); if(m==0) return 0; for(int i=1;i<=m;i++) scanf("%d",&lim[i]); sort(lim+1,lim+m+1);
		for(int i=1;i<=cnt;i++) for(int o=1;lim[o]<=i&&o<=m;o++) Link(i,i-lim[o]);
		for(int i=0;i<=cnt;i++) if(in[i]==0) q.push(i);
		while(!q.empty()){
			top=q.front(),q.pop(),mp.clear();
			for(int i:r[top]) mp.insert(g[i]);
			for(int i=0;i<=cnt+1;i++) if(!mp.count(i)){g[top]=i; break;}
			for(int i:k[top]) if(--in[i]==0) q.push(i);
		}
		scanf("%d",&T2); 
		while(T2--){scanf("%d",&T),x=0;  for(int i=1;i<=T;i++) scanf("%d",&num[i]),x^=g[num[i]]; putchar((x==0)?'L':'W');}		
		for(int i=0;i<=cnt;i++) k[i].clear(),r[i].clear(),g[i]=0,in[i]=0;
		putchar('\n');
		// printf("%d ms used\n",clock()-start);
		// if(clock()-start>15000) puts("TLE"),exit(0);
	}
	return 0;
}
```

---

第八题[P2599 取石子游戏](https://www.luogu.com.cn/problem/P2599)
这题和 nim 差的还真是挺多。。。
这个其实不看博弈他就是个 dp
however，1e9 的数据，1e3 的 n，dp 和记搜也属实是小有劣势。。。
我们还是从一个考虑
$[x]$ 一眼必胜

$[x,x]$ 一眼必败
$[x,y](x\neq y)$ 一眼必胜

$[x,x,y]$ 一眼必胜
$[x,y,x]$ 呢？
分析一下他必败
如果一次全取给对面的必胜
如果两人持续跟随则会变成 $[1,y,1]$
若 $y\neq 1$ 则显然必败
若 $y=1$ 则在上一步对方反手给一个 $(1,1)$
那么显然 $x,y,z$ 必胜
也就是说，首尾同败异胜

$[a,x,x,b]$ 显然必胜
$[a,x,x,a]$ 显然必败
$[x,x,y,x]$ 必胜。。。。

好像并没有啥规律
考虑进行 dfs

先思考部分分
显然一个状态应当是有开始位置，结束位置，首元素，尾元素组成
那大暴力还不会吗？？？

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef unsigned long long ull;
const int N=1000;
int n,k[N];
unordered_map<ull,bool> mp;
ull Hash(ull a,ull b,ull c,ull d){
	return ((a*10+b)*10000+c)*10000+d;
}
bool dfs(int lpos,int rpos,int lval,int rval){
	if(lpos>rpos) return 0;
	if(lpos==rpos) return 1; 
	if(lpos+1==rpos) return (lval!=rval);
	if(mp.count(Hash(lpos,rpos,lval,rval))) return mp[Hash(lpos,rpos,lval,rval)];
	mp[Hash(lpos,rpos,lval,rval)]=1;
	if(dfs(lpos+1,rpos,k[lpos+1],rval)==0) return 1;
	if(dfs(lpos,rpos-1,lval,k[rpos-1])==0) return 1;
	for(int i=1;i<lval;i++) if(dfs(lpos,rpos,i,rval)==0) return 1;
	for(int i=1;i<rval;i++) if(dfs(lpos,rpos,lval,i)==0) return 1;
	mp[Hash(lpos,rpos,lval,rval)]=0;
	return 0;
}
int main(){
	scanf("%d",&n);
	while(scanf("%d",&n)!=EOF) {
		mp.clear();
		for(int i=1;i<=n;i++) scanf("%d",&k[i]);
		printf("%d\n",dfs(1,n,k[1],k[n]));
	}
	return 0;
}
```

~~不要写了，也不要交了，一分没有~~

我们还是回顾一下最基本的知识

1. 全零必败
2. 每种状态非必胜即必败
3. 有必败后继的必胜
4. 无必败后继的必败

好了
接下来我们开题解 [Link](https://www.luogu.com.cn/article/ur4wk3gz)
简化一下

> 设 $L(i,j)$ 表示在 $[i,j]$ 区间的左侧放上一堆数量为 $L(i,j)$ 的石子后，**先手必败**。（$L(i,j)$ 可以为 $0$）  
> 即：$(L(i,j),a_i,a_{i+1},\cdots,a_j)$ 为必败局面。
> 同理对右侧定义 $R(i,j)$。
>
> 现证明其存在性和唯一性
>
> - 唯一性：若不唯一，则存在 $x_1,x_2(x_1 \neq x_2)$， $(x_1,a_i,a_{i+1},\cdots,a_{j-1},a_j)$ 和 $(x_2,a_i,a_{i+1},\cdots,a_{j-1},a_j)$ 均为必败局面。
>   而这两个必败局面之间实际一步可达，故矛盾，进而原命题成立。
> - 存在性：若不存在，则对于任意 $x\in \Z_+$，都满足 $(x,a_i,a_{i+1},\cdots,a_j)$ 必胜
>   则先手一步操作必取最右
>   而取一次后右侧未知
>   然而右侧有限取法左侧无限取法
>   显然存在 $x_1,x_2(x_1 \neq x_2),y$ 满足 $(x_1,a_i,a_{i+1},\cdots,a_{j-1},y)$ 和 $(x_2,a_i,a_{i+1},\cdots,a_{j-1},y)$ 都是必败局面。
>   但这两个必败局面之间实际一步可达，故矛盾，进而原命题成立。
>
> 根据上文定理二，如果放了其他数则定为必胜情况
>
> 接下来是转移
> 显然最简单的 $L(x,x)=R(x,x)=x$
> 接下来是普通情况 $L(x,y)$
>
> - 先判当前段是否必败，即 $R(x,y-1)=a_y$，必败的话填 $0$
> - 当 $a_y<\min\{L(x,y-1),R(x,y-1)\}$ ，有 $L(x,y)=a_y$
>   此时形如 $[a_y,a_x,a_{x+1},\dots,a_y]$
>   最小值情况后手持续跟随可得 $[k,a_x,a_{x+1},\dots,a_{y-1}]$ 或 $[a_x,a_{x+1},\dots,a_{y-1},k]$ 显然必胜
> - 当 $L(x,y-1)\le a_y\lt R(x,y-1)$ 时，有 $L(x,y)=a_y+1$
>    此时形如 $[a_y+1,a_x,a_{x+1},\dots,a_y]$
>   - 先手取左侧
>     若先手取左后大于 $L(x,y-1)$，则后手取同样数量进入递归
>     若先手取为 $L(x,y-1)$ ，则直接删除最后一列给先手一个必败
>     若先手取后小于该数就直接上跟随，最终同 Case 2 杀先手
>     若先手直接删一列有定义此为必胜形
>   - 先手取右侧
>     若先手取右后仍满足大于等于 $L(x,y-1)$ 则后手取同样数量进入递归
>     若先手取后小于该数就直接上跟随，最终同 Case 2 杀先手
>     若先手直接删一列有定义此为必胜形
> - 当 $R(x,y-1)\lt a_y\le L(x,y-1)$ 时，有 $L(x,y)=a_y-1$
>   此时形如 $[a_y-1,a_x,a_{x+1},\dots,a_y]$
>   - 先手取左侧
>     若先手取左后大于等于 $R(x,y-1)$，则后手取同样数量进入递归
>     若先手取后小于该数就直接上跟随，最终同 Case 2 杀先手
>     若先手直接删一列有定义此为必胜形
>   - 先手取右侧
>     若先手取左后大于 $R(x,y-1)$，则后手取同样数量进入递归
>     若先手取为 $R(x,y-1)$ ，则直接删除最左一列给先手一个必败
>     若先手取后小于该数就直接上跟随，最终同 Case 2 杀先手
>     若先手直接删一列有定义此为必胜形
> - 当 $a_y>\max\{L(x,y-1),R(x,y-1)\}$ ，有 $L(x,y)=a_y$
>   首先后手可以一直跟随递归
>   如果先手一把取到最小值以下就跟随同 Case 2
>   如果他取到所在的 $val$ 就删一列
>   如果在左取 $L(x,y-1)\lt x\le R(x,y-1)$ 就在右取 $x-1$ 同 Case 3
>   如果在左取 $R(x,y-1)\le x\lt L(x,y-1)$ 就在右取 $x+1$ 同 Case 4
>   右侧易得
>   如果直接删列显然必胜
>
> 那么有 
> $L(i,j)=
> \begin{cases}  
> 0, \quad &a_y=R(i,j-1),\\  
> a_y+1, \quad &L(i,j-1) \le a_y < R(i,j-1),\\  
> a_y-1, \quad &R(i,j-1) < a_y \le L(i,j-1),\\  
> a_y, \quad &\text{otherwise}.\\
> \end{cases}
> \qquad
> R(i,j)=
> \begin{cases}  
> 0, \quad &a_x=L(i+1,j),\\  
> a_x-1, \quad &L(i+1,j) \lt a_x \le R(i+1,j),\\  
> a_x+1, \quad &R(i+1,j) \le a_x \lt L(i+1,j),\\  
> a_x, \quad &\text{otherwise}.\\
> \end{cases}
> $
> 写法类似区间 dp
>
> ```cpp
> #include<bits/stdc++.h>
> using namespace std;
> const int N=1009;
> int n,a[N],L[N][N],R[N][N];
> int main(){
>     int T; scanf("%d",&T);
>     while(T--){
>         scanf("%d",&n); for(int i=1;i<=n;i++) scanf("%d",&a[i]),L[i][i]=R[i][i]=a[i];
>         for(int len=1;len<=n;len++) for(int i=1,j;i+len-1<=n;i++) {
>             j=i+len-1;
>             if(R[i][j-1]==a[j]) L[i][j]=0;
>             else if(a[j]<R[i][j-1]&&a[j]>=L[i][j-1]) L[i][j]=a[j]+1;
>             else if(a[j]>=R[i][j-1]&&a[j]<L[i][j-1]) L[i][j]=a[j]-1;
>             else L[i][j]=a[j];
>             if(L[i+1][j]==a[i]) R[i][j]=0;
>             else if(a[i]<=R[i+1][j]&&a[i]>L[i+1][j]) R[i][j]=a[i]-1;
>             else if(a[i]>=R[i+1][j]&&a[i]<L[i+1][j]) R[i][j]=a[i]+1;
>             else R[i][j]=a[i];
>         }
>         if(L[2][n]==a[1]) puts("0"); else puts("1");
>     }
>     return 0;
> }
> ```

~~润题解就到这里~~
但这是题解。。。
怎么正推呢？
显然最简单的 dp 策略是类似区间 dp
设一个布尔数组表示某段是否必胜
然而我们发现转移不好办（因为信息几乎不可合并）
所以想到这种抽象方式
再加上考场现搞关于存在性和唯一性的证明

假设我们 dp 框架已经有了
然后思考转移
对于必败序列显然为零

对于其他情况呢
分类讨论是必要的
但是我们一开始可能并不能分出这么好的
所以我们从头来分

|  限制   | $a_y>R$ | $a_y<R$ |
| :-----: | :-----: | :-----: |
| $a_y>L$ |         |         |
| $a_y=L$ |         |         |
| $a_y<L$ |         |         |

突破口应当是双小于
我们发现如果不算当前列他就是一个必胜，后手可以选择跟随取胜

余下同理


###  例题

#### [P1463 反素数](https://www.luogu.com.cn/problem/P1463)

这题显然数字个数不多可以打表过
用前面的筛法会炸空间
那我们可以思考直接打表

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=50000;
int p[N],cnt;
bool st[N];
void eular(int n=49980){
	for(int i=2;i<=n;i++){
		if(!st[i])p[++cnt]=i;
		for(int j=1;j<=cnt&&p[j]<=n/i;j++){st[i*p[j]]=1;if(i%p[j]==0)break;}
	}
}
int main(){
	int n,big=1,eps=2;
	scanf("%d",&n),puts("1"),eular();
	for(int i=2,tmp=1,j;i<=n;i+=eps){
		j=i,tmp=1;
		for(int k=1,t;k<=cnt&&p[k]<j;k++){
			t=1;
			while(j%p[k]==0) j/=p[k],++t;
			tmp*=t;
		}
		if(j>1)tmp*=2;
		if(tmp>big)big=tmp,printf("%d\n",i);
	}
	return 0;
}
```

打了个这样的表

```cpp
1
2
4
6
12
24
36
48
60
120
180
240
360
720
840
1260
1680
2520
5040
7560
10080
15120
20160
25200
27720
45360
50400
55440
83160
110880
166320
221760
277200
332640
498960
554400
665280
720720
```

我们发现 $720$ 以后也可以每次加 $120$

```cpp
1
2
4
6
12
24
36
48
60
120
180
240
360
720
840
1440
1680
2520
5040
7560
10080
15120
20160
25200
27720
45360
50400
55440
83160
110880
166320
221760
277200
332640
498960
554400
665280
720720
1081080
1441440
2162160
2882880
3603600
4324320
6486480
7207200
8648640
10810800
14414400
17297280
21621600
32432400
36756720
43243200
61261200
73513440
110270160
122522400
147026880
183783600
245044800
294053760
367567200
551350800
698377680
735134400
1102701600
1396755360
QAQ
```

大概跑了不到五分钟

```cpp
#include<bits/stdc++.h>
using namespace std;
int ans[509]={1,2,4,6,12,24,36,48,60,120,180,240,360,720,840,1440,1680,2520,5040,7560,10080,15120,20160,25200,27720,45360,50400,55440,83160,110880,166320,221760,277200,332640,498960,554400,665280,720720,1081080,1441440,2162160,2882880,3603600,4324320,6486480,7207200,8648640,10810800,14414400,17297280,21621600,32432400,36756720,43243200,61261200,73513440,110270160,122522400,147026880,183783600,245044800,294053760,367567200,551350800,698377680,735134400,1102701600,1396755360};
int main(){
	int n;
	scanf("%d",&n);
	printf("%d",ans[upper_bound(ans,ans+68,n)-ans-1]);
	return 0;
}
```

#### [P1072  Hankson](https://www.luogu.com.cn/problem/P1072)

这题简单思路直接枚举 $a_1$ 倍数上界 $b_1$
极限复杂度 $10^{14}$ 过不了
注意到他同时需要是 $\frac{b_1}{b_0}$ 倍数
$\text{TLE 80pts Code}$

```cpp{
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll GCD(ll a,ll b){return (b==0)?a:GCD(b,a%b);}
ll LCM(ll a,ll b){return a/GCD(a,b)*b;}
int main(){
	int n,ans;
    ll a,b,c,d,t;
	scanf("%d",&n);
	while(n--){
        scanf("%lld%lld%lld%lld",&a,&b,&c,&d),t=LCM(b,d/c),ans=0;
        for(int i=t;i<=d;i+=t) if(GCD(i,a)==b&&LCM(i,c)==d) ++ans;
        printf("%d\n",ans);
    }
	return 0;
}
```

发现另一个性质
即它必然需要是 $d$ 的约数
那我们只需要枚举到根号即可

$\text{AC 100pts Code}$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll GCD(ll a,ll b){return (b==0)?a:GCD(b,a%b);}
ll LCM(ll a,ll b){return a/GCD(a,b)*b;}
int main(){
	int n,ans;
    ll a,b,c,d,t;
	scanf("%d",&n);
	while(n--){
        scanf("%lld%lld%lld%lld",&a,&b,&c,&d),ans=0;
        for(int i=1;i*i<=d;i++){
        	if(d%i!=0)continue;
        	if(GCD(i,a)==b&&LCM(i,c)==d) ++ans;
        	if(i*i==d)break;
        	if(GCD(d/i,a)==b&&LCM(d/i,c)==d) ++ans;
        } 
        printf("%d\n",ans);
    }
	return 0;
}
```

#### [LibreOJ 10205. 最大公约数](https://loj.ac/p/10205)

更相减损

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef vector<char> num;
void pop(num &x){while(x[x.size()-1]==0) x.pop_back();}
void scan(num &x,char *k){x.clear(); for(int i=strlen(k)-1;i>=0;i--) x.push_back(k[i]-'0'); pop(x);}
void print(num x){for(int i=x.size()-1;i>=0;i--) putchar(x[i]+'0'); putchar('\n');}
bool operator <(num a,num b){
	if(a.size()<b.size()) return 1;
	if(a.size()>b.size()) return 0;
	for(int i=a.size()-1;i>=0;i--) if(a[i]<b[i])return 1; else if(a[i]>b[i])return 0;
	return 0;
}
bool operator ==(num a,num b){
	if(a.size()!=b.size()) return 0;
	for(int i=a.size()-1;i>=0;i--) if(a[i]!=b[i]) return 0;
	return 1;
}
num operator -(num a,num b){
	num c=a;
	int len=b.size();
	for(int i=0;i<len;i++) if(c[i]>=b[i])c[i]-=b[i]; else c[i+1]--,c[i]+=10-b[i];
	for(int i=len;i<c.size();i++) if(c[i]<0) c[i]+=10,c[i+1]--; else break;
	pop(c);
	return c;
}
void Double(num &x){
	int len=x.size()-1;
	for(int i=0;i<=len;i++) x[i]<<=1;
	for(int i=0;i<len;i++) x[i+1]+=x[i]/10,x[i]%=10;
	while(x[len]>=10) x.push_back(x[len]/10),x[len++]%=10;
}
void Half(num &x){
	int cnt=0,it=x.size()-1;
	for(int tmp;it>=0;it--){
		cnt*=10,tmp=cnt,cnt=x[it]&1;
		x[it]=(tmp+x[it])>>1;
	}
	pop(x);
}
num GCD(num a,num b){
	int cnt=0;
	num tmp;
	while(true){
		while(a[0]%2==0&&b[0]%2==0) ++cnt,Half(a),Half(b);
		if(a==b){while(cnt--) Double(a); return a;}
		if(a<b) swap(a,b);
		tmp=a,a=b,b=tmp-b;
	}
}
char s1[3004],s2[3004];
num a,b,c;
int main(){
	scanf("%s%s",s1,s2),scan(a,s1),scan(b,s2),c=GCD(a,b),print(c);
	return 0;
}
```

火车头优化可过

#### [LibreOJ 10206. X-factor Chain](https://loj.ac/p/10206)

~~终于来水题啦！~~
这题对原数分解质因数，然后搞一个组合数即可
~~仔细思考发现用不上 ULL~~

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=1111,n=1109;
bool st[N];
int p[N],cnt;
void eular(){
    for(int i=2;i<=n;i++){
        if(!st[i]) p[++cnt]=i;
        for(int j=1;j<=cnt&&p[j]<=n/i;j++){st[i*p[j]]=1; if(i%p[j]==0) break;}
    }
}
unsigned long long ans2=1;
int main(){
    int x,ans1=0,it=0,tmp=0;
    scanf("%d",&x),eular;
    for(int i=1;p[i]*p[i]<=x;i++,tmp=0){
        while(x%p[i]) ++tmp,ans2*=++it,x/=p[i];
        ans1+=tmp;
        for(int j=1;j<=tmp;j++) ans2/=j;
    }
    printf("%d %llu\n",ans1,ans2);
    return 0;
}
```

#### [P4397 聪明的燕姿](https://www.luogu.com.cn/problem/P4397)

思路很好想
我们逆用约数和解析式不就好了？
我们首先要把那个数分解质因数
然后呢？暴力搞出每一个约数？好像搞一个 $map$ 也可做。。。
貌似搞一手预处理二次以上的？好像也不多可以行。。。
然后呢？爆搜 DFS？。。。

~~看了个 TJ~~
搜索是没问题的，但思路很重要
原来那个被我一通乱搞已经几乎不可做了。。。
我们直接去搜原数
每次的 $dfs$ 维护还可以分解的数字，当前的质数个数（避免重复）以及 $\text{current ans}$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=44721,M=1000003;
int p[N],cnt,ans[M],idx;
bool st[N+5];
void eular(){
    for(int i=2;i<=N;i++){
        if(!st[i]) p[++cnt]=i;
        for(int j=1;j<=cnt&&p[j]<=N/i;j++) {st[i*p[j]]=1; if(i%p[j]==0) break;}
    }
}
bool check(int x){
	for(int i=2;i*i<=x;i++) if(x%i==0) return 0;
	return 1;
}
void dfs(int x,int np,int k){
    if(x==1){ans[++idx]=k;return;}
    for(int i=np;i<=cnt&&p[i]+1<=x;i++) for(ll rt=p[i]+1,t=p[i];rt<=x;rt=rt*p[i]+1,t*=p[i]) if(x%rt==0) dfs(x/rt,i+1,k*t);
    if(x-1>N&&check(x-1)) ans[++idx]=k*(x-1);
}
signed main(){
    eular();int x;
    while(scanf("%d",&x)!=EOF){
        idx=0,dfs(x,1,1);
        printf("%d\n",idx);
        if(idx>0){
            sort(ans+1,ans+idx+1);
            for(int i=1;i<=idx;i++) printf("%d ",ans[i]);
            putchar('\n');
        }
    }
}
```

这个只能过 $\text{luogu}$
LOJ 和 ACC 上都会 TLE
思考如果当前的 $x$ 是某个质数加上一
那么他的一次分解是唯一的
判断可以直接到 $\sqrt n$

要是并非的话那么就一定满足他至少有两个质因子
其中至少有一个小于 $\sqrt{n}$
而另外的因子可以在以后的递归中搞
注意要比当前最小质因子大才能统计

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=44721,M=1000003;
int p[N],cnt,ans[M],idx;
bool st[N+5];
void eular(){
    for(int i=2;i<=N;i++){
        if(!st[i]) p[++cnt]=i;
        for(int j=1;j<=cnt&&p[j]<=N/i;j++) {st[i*p[j]]=1; if(i%p[j]==0) break;}
    }
}
bool check(int x){
	for(int i=2;i*i<=x;i++) if(x%i==0) return 0;
	return 1;
}
void dfs(int x,int np,int k){
    if(x==1){ans[++idx]=k;return;}
    for(int i=np;i<=cnt&&p[i]*p[i]+p[i]+1<=x;i++) for(ll rt=p[i]+1,t=p[i];rt<=x;rt=rt*p[i]+1,t*=p[i]) if(x%rt==0) dfs(x/rt,i+1,k*t);
    if(x-1>=p[np]&&check(x-1)) ans[++idx]=k*(x-1);
}
signed main(){
    eular();int x;
    while(scanf("%d",&x)!=EOF){
        idx=0,dfs(x,1,1);
        printf("%d\n",idx);
        if(idx>0){
            sort(ans+1,ans+idx+1);
            for(int i=1;i<=idx;i++) printf("%d ",ans[i]);
            putchar('\n');
        }
    }
}
```



#### [P2152 SuperGCD](https://www.luogu.com.cn/problem/P2152)

更相减损法的加强版

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef vector<char> num;
inline void pop(num &x){while(x[x.size()-1]==0) x.pop_back();}
void scan(num &x,char *k){x.clear(); for(int i=strlen(k)-1;i>=0;i--) x.push_back(k[i]-'0'); pop(x);}
void print(num x){for(int i=x.size()-1;i>=0;i--) putchar(x[i]+'0'); putchar('\n');}
bool operator <(num a,num b){
	if(a.size()<b.size()) return 1;
	if(a.size()>b.size()) return 0;
	for(int i=a.size()-1;i>=0;i--) if(a[i]<b[i])return 1; else if(a[i]>b[i])return 0;
	return 0;
}
bool operator ==(num a,num b){
	if(a.size()!=b.size()) return 0;
	for(int i=a.size()-1;i>=0;i--) if(a[i]!=b[i]) return 0;
	return 1;
}
num operator -(num a,num b){
	num c=a;
	int len=b.size();
	for(int i=0;i<len;i++) if(c[i]>=b[i])c[i]-=b[i]; else c[i+1]--,c[i]+=10-b[i];
	for(int i=len;i<c.size();i++) if(c[i]<0) c[i]+=10,c[i+1]--; else break;
	pop(c);
	return c;
}
void Double(num &x){
	int len=x.size()-1;
	for(int i=0;i<=len;i++) x[i]<<=1;
	for(int i=0;i<len;i++) x[i+1]+=x[i]/10,x[i]%=10;
	while(x[len]>=10) x.push_back(x[len]/10),x[len++]%=10;
}
void Half(num &x){
	int cnt=0,it=x.size()-1;
	for(int tmp;it>=0;it--){
		cnt*=10,tmp=cnt,cnt=x[it]&1;
		x[it]=(tmp+x[it])>>1;
	}
	pop(x);
}
num GCD(num a,num b){
	int cnt=0;
	num tmp;
	while(true){
		while(a[0]%2==0&&b[0]%2==0) ++cnt,Half(a),Half(b);
		while(a[0]%2==0) Half(a); while(b[0]%2==0) Half(b);
		if(a==b){while(cnt--) Double(a); return a;}
		if(a<b) swap(a,b);
		tmp=a,a=b,b=tmp-b;
	}
}
char s1[30004],s2[30004];
num a,b,c;
int main(){
	scanf("%s%s",s1,s2),scan(a,s1),scan(b,s2),c=GCD(a,b),print(c);
	return 0;
}
```

#### [LibreOJ 10209. 青蛙的约会](https://loj.ac/p/10209)

$exgcd$ 模板题
$(km+x-kn-y)\%L=0 \to a(m-n)+bL=y-x$
要求两数之差能被 $GCD(m-n,L)$ 整除
负数可能需要特判
用到了求最小正整数解的方法

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

ll exgcd(ll a,ll b,ll &x,ll &y){
    if(b==0) {x=1,y=0;return a;}
    int ans=exgcd(b,a%b,y,x);
    y-=a/b*x;
    return ans;
}
int main(){
    ll m,n,x,y,L,ans,a,b,tmp;
    scanf("%lld%lld%lld%lld%lld",&x,&y,&m,&n,&L);
    if(x>y) swap(x,y),swap(m,n);
    if(m<n) swap(m,n),ans=-1;
    tmp=exgcd(m-n,L,a,b);
    if((y-x)%tmp!=0) puts("Impossible"),exit(0);
    else printf("%lld\n",((y-x)/tmp*a*ans%L+L)%L);
    return 0;
}
```

#### [LibreOJ 2605. 同余方程](https://loj.ac/p/2605)

逆元板子

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef ll long long;
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){a=1,b=0; return a;} int ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
int main(){
    ll a,b,x,y,ans;
    scanf("%lld%lld",&a,&b),exgcd(a,b,x,y),ans=(x%b+b)%b,printf("%lld",(ans!=0)?ans:b);
    return 0;
}
```

#### [LibreOJ 10211. Sumdiv](https://loj.ac/p/10211)

思考暴力

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll a,b,ans=1; const ll m=9901;
int main(){
    scanf("%lld%lld",&a,&b);
    for(int i=1;i<=b;i++) ans=(ans*a+1)%m;
    printf("%lld\n",ans);
}
```

发现此方式仅对质数有效
那就分解质因数再乘
加一个费马小定理优化

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll m=9901,N=50009;
bool st[N]; int p[N],cnt;
void eular(int n=50000){
	for(int i=2;i<=n;i++){
		if(!st[i]) p[++cnt]=i;
		for(int j=1;j<=cnt&&p[j]<=n/i;j++){st[i*p[j]]==0; if(i%p[j]==0) break;}
	}
}
inline ll tmp(ll a,ll b){b%=m-1;ll ans=1; for(int i=1;i<=b;i++) ans=(ans*a+1)%m; return ans;}
int main(){
	ll a,b,ans=1; 
    scanf("%lld%lld",&a,&b),eular();
    for(int i=1,tmpb;p[i]*p[i]<=a;i++){
    	tmpb=0;
    	while(a%p[i]==0) a/=p[i],tmpb+=b;
    	if(tmpb) ans=ans*tmp(p[i],tmpb)%m;
    } 
    if(a>1) ans=ans*tmp(a,b)%m;
    printf("%lld\n",ans);
}
```

#### [P2485 计算器](https://www.luogu.com.cn/problem/P2485)

三个板子

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll Pow(ll a,ll x,ll p){
	ll ans=1,tmp=a%p;
    while(x){
        if(x&1) ans=ans*tmp%p;
        tmp=tmp*tmp%p,x>>=1;
    }
    return ans;
}
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){x=1,y=0;return a;} ll ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
map<ll,ll> mp;
int main(){
    int T,type;
    ll a,b,c,tmp,x,y,A,B;
    scanf("%d%d",&T,&type);
    while(T--){
        scanf("%lld%lld%lld",&a,&b,&c);
        if(type==1) printf("%lld\n",Pow(a,b,c));
        else if(type==2){
            tmp=exgcd(a,c,x,y);
            if(b%tmp!=0) puts("Orz, I cannot find x!"); else x*=b/tmp,tmp=c/tmp,printf("%lld\n",(x%tmp+tmp)%tmp);
        }else{
            mp.clear(),a%=c,b%=c,A=sqrt(c)+1,B=Pow(a,A,c);
            if(b==1){puts("0"); continue;}
            else if(a==0&&b!=0){puts("Orz, I cannot find x!"); continue;}
            else if(a==0&&b==0){puts("1"); continue;}
            for(ll i=0,t=b;i<A;i++,t=t*a%c) mp[t]=i;
            for(ll i=1,t=B;i<=A+1;i++,t=t*B%c) if(mp.count(t)){printf("%lld\n",i*A-mp[t]); goto here;}
            puts("Orz, I cannot find x!");
        }
        here:;
    }
    return 0;
}
```

#### [P2421 荒岛野人](https://www.luogu.com.cn/problem/P2421)

貌似暴力枚举……
枚举共有多少个山洞
开一个哈希表记录每一轮所有的野人位置
暴力模拟时间复杂度 $O(nML)\to10^{14}$

枚举山洞是对的
设当前有 $k$ 个山洞
$$
(c_i+xp_i)\%k\neq(c_j+xp_j)\%k \Rightarrow x(p_i-p_j)+yk=c_j-c_i\ (x<=\min{(l_i,l_j)})\ \texttt{has no solution.}
$$
这个用 exGCD 可以暴力检查
时间复杂度 $O(n^2M\log M)$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=19;
int exgcd(int a,int b,int &x,int &y){if(b==0){x=1,y=0; return a;} int ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
int n,c[N],p[N],l[N],lim,tmp,x,y;
int main(){
    scanf("%d",&n),lim=n; for(int i=1;i<=n;i++) scanf("%d%d%d",&c[i],&p[i],&l[i]),lim=max(lim,c[i]); --lim;
   	while(++lim){
        for(int i=1;i<n;i++) for(int j=i+1;j<=n;j++){
        	if(p[i]==p[j]) continue;
            tmp=exgcd(p[i]-p[j],lim,x,y),x*=(c[j]-c[i])/tmp;
            if((c[j]-c[i])%tmp!=0) continue;
            tmp=abs(lim/tmp),x=(x%tmp+tmp)%tmp;
            if(x<=min(l[i],l[j])) goto fail;
        }
        printf("%d\n",lim),exit(0); fail:;
    }
}
```

#### [LibreOJ 10216. 五指山](https://loj.ac/p/10216)

exGCD 板子
要开 `long long`

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
int exgcd(int a,int b,int &x,int &y){if(b==0){x=1,y=0; return a;} int ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
signed main(){
    int T,n,d,l,r,x,y,tmp; scanf("%lld",&T);
    while(T--){
        scanf("%lld%lld%lld%lld",&n,&d,&l,&r),tmp=exgcd(d,n,x,y),x*=(r-l)/tmp;
        if((r-l)%tmp!=0) puts("Impossible");
        else tmp=n/tmp,printf("%lld\n",(x%tmp+tmp)%tmp);
    }
    return 0;
}
```

#### [LibreOJ 10217. Biorhythms](https://loj.ac/p/10217)

中国剩余定理板子题
可以先计算一些
$x\%23=a,x\%28=b,x\%33=c$
$23*28=644,23*33=759,28*33=924$
$inv(644)=2,inv(759)=19,inv(924)=6$

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    long long a,b,c,d,ans;
    while(scanf("%lld%lld%lld%lld",&a,&b,&c,&d))
        if(a<0) return 0;
        else ans=((c*644*2+b*759*19+a*924*6-d)%21252+21252)%21252,printf("%lld\n",(ans==0)?21252:ans);
    return 0;
}
```

#### [LibreOJ 10218. C Loops](https://loj.ac/p/10218)

exGCD 板子题

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll exgcd(ll a,ll b,ll &x,ll &y){if(b==0){x=1,y=0; return a;} ll ans=exgcd(b,a%b,y,x); y-=a/b*x; return ans;}
int main(){
    ll a,b,c,k,t,x,y;
    while(scanf("%lld%lld%lld%lld",&a,&b,&c,&k)){
    	if(k==0) return 0;
        k=(1ll<<k),b=((b-a)%k+k)%k,t=exgcd(c,k,x,y),x*=b/t;
        if(b%t!=0) puts("FOREVER"); else t=k/t,printf("%lld\n",(x%t+t)%t);
    }
}
```

#### [LibreOJ 10221. 斐波那契求和](https://loj.ac/p/10221)

矩乘模板题
定义基础矩阵 $\begin{bmatrix}a_{i-1}&a_i&\sum^i_{it=1}{a_{it}}\end{bmatrix}$
容易构造出矩阵 
$$
\begin{bmatrix}
0 & 1 & 1\\
1 & 1 & 1\\
0 & 0 & 1
\end{bmatrix}
$$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=4;
ll Mod;
template<class T> struct matrix{
		friend matrix operator *(matrix a,matrix b){
			matrix c;
			int n=a.n,m=b.n,p=b.m;
			c.n=n,c.m=p;
			for(int i=1;i<=n;i++) for(int j=1;j<=p;j++) for(int it=1;it<=m;it++) 
                c.k[i][j]=(c.k[i][j]+a.k[i][it]*b.k[it][j])%Mod;
			return c;
		}
		matrix(){n=m=0;}
		matrix(int _n,int _m){n=_n,m=_m;}
		matrix(int _n,int _m,T** x){
			n=_n,m=_m;
			for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) k[i][j]=x[i][j];
		}
		T k[N][N]={};
		int n,m;
};
matrix<long long> A(1,3),B(3,3);
template<class T> matrix<T> Pow(matrix<T> base,ll x){
    matrix<T> ans(3,3);
    ans.k[1][1]=ans.k[2][2]=ans.k[3][3]=1;
    while(x){
        if(x&1) ans=ans*base;
        x>>=1,base=base*base;
    }
    return ans;
}
int main(){
    ll n;
    scanf("%lld%lld",&n,&Mod);
    if(n<2) putchar('1'),exit(0);
    A.k[1][1]=A.k[1][2]=1,A.k[1][3]=2;
    B.k[1][2]=B.k[1][3]=B.k[2][1]=B.k[2][2]=B.k[2][3]=B.k[3][3]=1;
    printf("%lld\n",(A*Pow(B,n-2)).k[1][3]);
    return 0;
}
```

#### [LibreOJ 10222. 佳佳的数列](https://loj.ac/p/10222)

我们搞斐波那契是容易的
只是不好乘上一个变化的数
考虑计算 $n(F_1+F_2+\dots+F_n)-\sum_{i=1}^n(n-i)F_i$
定义后一项为 $S(n)$ 则有公式 $S(i)=S(i-1)+Sum(i-1)$
容易得出基础矩阵为  $\begin{bmatrix}a_{i-1}&a_i&sum&S(i)\end{bmatrix}$
而构造矩阵如下
$$
\begin{bmatrix}
0 & 1 & 1 & 0\\
1 & 1 & 1 & 0\\
0 & 0 & 1 & 1\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=6;
ll Mod;stack<char> st;
template<class T> struct matrix{
		friend matrix operator *(matrix a,matrix b){
			matrix c;
			int n=a.n,m=b.n,p=b.m;
			c.n=n,c.m=p;
			for(int i=1;i<=n;i++) for(int j=1;j<=p;j++) for(int it=1;it<=m;it++) 
                c.k[i][j]=(c.k[i][j]+a.k[i][it]*b.k[it][j])%Mod;
			return c;
		}
		matrix(){n=m=0;}
		matrix(int _n,int _m){n=_n,m=_m;}
		matrix(int _n,int _m,T** x){
			n=_n,m=_m;
			for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) k[i][j]=x[i][j];
		}
		T k[N][N]={};
		int n,m;
		void print(){for(int i=1;i<=n;i++){for(int j=1;j<=m;j++) _print(k[i][j]),putchar(' '); putchar('\n');} }
		void _print(T x){
			if(x<0) putchar('-'),x=-x;
			do st.push(x%10+'0'),x/=10; while(x);
			while(!st.empty()) putchar(st.top()),st.pop();
		}
};
matrix<ll> A(1,4),B(4,4),C(1,4);
template<class T> matrix<T> Pow(matrix<T> base,ll x){
    matrix<T> ans(4,4);
    ans.k[1][1]=ans.k[2][2]=ans.k[3][3]=ans.k[4][4]=1;
    while(x){
        if(x&1) ans=ans*base;
        x>>=1,base=base*base;
    }
    return ans;
}
int main(){
    ll n;
    scanf("%lld%lld",&n,&Mod);
    if(n==1) putchar('1'),exit(0);
    if(n==2) putchar('3'),exit(0);
    A.k[1][1]=A.k[1][2]=A.k[1][4]=1,A.k[1][3]=2;
    B.k[1][2]=B.k[1][3]=B.k[2][1]=B.k[2][2]=B.k[2][3]=B.k[3][3]=B.k[3][4]=B.k[4][4]=1;
    C=A*Pow(B,n-2),printf("%lld\n",((n%Mod*C.k[1][3]%Mod-C.k[1][4])%Mod+Mod)%Mod);
    return 0;
}
```

#### [P3193 GT考试](https://www.luogu.com.cn/problem/P3193)

##### 错误思路
直接找含数字段的个数，然后用总数减去
找的方法直接固定位置（如 $\texttt{XX0219XXXXX}$）然后快速幂搞一下
然而存在如 $\texttt{XX0219X0219}$ 这样的数字会被多次计算
从而导致我们最终结果错误

##### 优化思路

自然想到设 $dp_i$ 表示在长度为 $i$ 时的答案
如果能推导出式子可能可以矩乘优化
然而原数字可能重叠（如对于数段 $\texttt{101}$）
我们只有一维很不好搞

~~润题解~~发现这题设 $dp_{i,j}$ 为示长串匹配到第 $i$ 位，短串匹配到第 $j$ 位的方案数
然后我们发现每一个 $dp_{i,j}$ 都有上次循环中许多数字转移而来
而他们的具体关系很复杂不好直接表示
于是我们设一个辅助数组 $g(i,j)$ 表示由 $dp_{x,i}$ 转移向 $dp_{x,j}$ 的方案数（就是结尾加上的数字）

##### $g$ 函数的必要性

起初我觉的这个方式相当愚蠢
那能加几个？肯定要么零要么一啊！
然而一方面这个"要么零要么一"也无法直接搞出来
何况还有一个逆天大雷 $dp_{x,0}$
所以这个设计是有其必要性的

##### 整体步骤

接下来就简单了
我们一共三步

1. 求出 $g$ 函数的映射
1. 矩乘优化转移 $dp_{i,j}=\sum g(j',j)\times dp_{i-1,j'}$

##### Step 1

一眼 KMP
然后我们枚举来源和加入的数字
再持续跳 $next$ 进行累加答案

```cpp
const int N=25;
int nxt[N],g[N][N];
void kmp(int l,char *s){
    nxt[0]=-1;
    for(int i=1,j=-1;i<=l;i++){
        while(~j&&s[j+1]!=s[i]) j=nxt[j];
        nxt[i]=++j;
    }
    for(int i=1,j=0;i<=l;i++) for(char c='0';c<='9';c++){
        j=i-1;//匹配数量
        while(~j&&s[j+1]!=c) j=nxt[j];
        ++g[i-1][++j];
    }
    g[l][l]=10;//把不合法的全判掉
}
```

##### Step 2

我们思考发现只用到上一层的所有 $dp$ 值
容易想到把它们设成一个矩阵
我们先设 $A_{1,n}$
显然需要右乘一个 $n$ 阶方阵
不难构造矩阵
$$
A=
\begin{bmatrix}
g(0,0) & g(0,1) & \cdots & g(0,n)\\
g(1,0) & g(1,1) & \cdots & g(1,n)\\
\vdots & \vdots & \ddots & \vdots\\
g(n,0) & g(n,1) & \cdots & g(n,n)
\end{bmatrix}
,B=
\begin{bmatrix}
9 & 1 & 0 & \cdots & 0\\
\end{bmatrix}
$$
然后我们直接矩乘快速幂 $B*A^{n-1}$

##### Code

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=25;
int n,m,Mod,nxt[N];
char s[N];
stack<char> st;
template <class T> struct matrix{
    friend matrix operator *(matrix a,matrix b){
    	matrix c(a.n,b.m);
    	for(int i=1;i<=a.n;i++) for(int j=1;j<=b.m;j++) for(int it=1;it<=b.m;it++) 
            c.k[i][j]=(a.k[i][it]*b.k[it][j]%Mod+c.k[i][j])%Mod;
		return c; 
	}
	void basic(int size){
		n=m=size;
		for(int i=1;i<=n;i++) for(int j=1;j<=n;j++) k[i][j]=0;
		for(int i=1;i<=n;i++) k[i][i]=1;
	}
	matrix(){n=m=0;}
	matrix(int _n,int _m){n=_n,m=_m;}
	matrix(int _n,int _m,T *x){n=_n,m=_m,k=x;}
    int n,m;
    T k[N][N]={};
};
template<class T> matrix<T> Pow(matrix<T> x,int d){
	matrix<T> ans;
	ans.basic(x.n);
	while(d){
		if(d&1) ans=ans*x;
		x=x*x,d>>=1;
	}
	return ans;
}
int main(){
	int ans=0;
	scanf("%d%d%d%s",&n,&m,&Mod,s+1);
	matrix<int> A(m+1,m+1),B(1,m+1),tmp;
	B.k[1][1]=9,B.k[1][2]=1,nxt[0]=-1;
	for(int i=1,j=-1;i<=m;i++){
		while(~j&&s[j+1]!=s[i]) j=nxt[j];
		nxt[i]=++j;
	}
	for(int i=1,j=0;i<=m;i++) {
		for(char c='0';c<='9';c++){
			j=i-1;
			while(~j&&s[j+1]!=c) j=nxt[j];
			A.k[i][j+2]++; //对于 i-1 j+1 的补偿  
		}
	}
	A.k[m+1][m+1]=10;
	tmp=B*Pow(A,n-1);
	for(int i=1;i<=m;i++) ans=(ans+tmp.k[1][i])%Mod;
	printf("%d\n",ans);
	return 0;
}
```

#### [P2480 古代猪文](https://www.luogu.com.cn/problem/P2480)

根据费马小定理我们指数要对于模数减一取模
那我们就把模数分解质因数，Lucas 统计答案，CRT 求答案即可
$$
\text{ans}=g^{\sum_{k|n} C^k_n}\mod 999911659=g^{\sum_{k|n} C^k_n\bmod 999911658}\bmod 999911659
\\
999911658=2*3*4679*35617
$$

我们只需要对这四个数分别统计其答案
得到
$$
\begin{cases}
ans\bmod 2=a_1\\
\dots\\
ans\bmod 35617=a_4
\end{cases}
$$
用 CRT 搞一下即可

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=36000,num[4]={2,3,4679,35617};
const ll Mod=999911658;
unordered_set<int> mp;
int inv[N],fact[N],ans[4];
ll ANS;
inline int C(int n,int m,int p){if(m>n) return 0; else return fact[n]*inv[fact[n-m]]%p*inv[fact[m]]%p;}
int Lucas(int n,int m,int p){
	if(n<p) return C(n,m,p);
	else return Lucas(n/p,m/p,p)*C(n%p,m%p,p)%p;
}
int Pow(ll n,int m,int p){
	int t=1;
	while(m){
		if(m&1) t=t*n%p;
		m>>=1,n=n*n%p;
	}
	return t;
}
int Inv(int n,int p){return Pow(n,p-2,p);}
int main(){
	int n,g; scanf("%d%d",&n,&g);
	for(int i=1;i*i<=n;i++) if(n%i==0) mp.insert(i),mp.insert(n/i);
	for(int i=0,p;i<4;i++){
		p=num[i],inv[1]=fact[0]=fact[1]=1; for(int j=2;j<=p;j++) inv[j]=(-p/j*inv[p%j]%p+p)%p,fact[j]=fact[j-1]*j%p;
		for(int x:mp) ans[i]=(ans[i]+Lucas(n,x,p))%p;
	}
	for(int i=0;i<4;i++) ANS=(ANS+Mod/num[i]*Inv(Mod/num[i],num[i])%Mod*ans[i]%Mod)%Mod;
	printf("%d\n",Pow(g,ANS,Mod+1));
	return 0;
}
```

这题其实还有一个  Hack 数据
就是一种特殊情况：当 ANS 为 Mod 倍数的时候
这种情况下正常就应该是一
然而，有可能 $g$ 是 Mod 的倍数而由于我们没算就输出了一
而实际上应当是零
那我们就把指数加上一个 Mod 就行了

#### [LibreOJ 10232. 车的放置](https://loj.ac/p/10232)

思考枚举上下各放置几个车
首先乘上一个选择行数的方案数 $C^k_n$
然后在乘上选择的排列数 $A^k_n$

```cpp
#include<bits/stdc++.h>
using namespace std;
const int p=100003;
int inv[p+3],fact[p+3],a,b,c,d,e,ans=0;
int C(int n,int m){if(n<m) return 0; return 1ll*fact[n]*inv[fact[n-m]]%p*inv[fact[m]]%p;}
int A(int n,int m){if(n<m) return 0; return 1ll*fact[n]*inv[fact[n-m]]%p;}
int main(){
    inv[1]=fact[0]=fact[1]=1; for(int i=2;i<p;i++) fact[i]=1ll*fact[i-1]*i%p,inv[i]=(-1ll*p/i*inv[p%i]%p+p)%p;
    scanf("%d%d%d%d%d",&a,&b,&c,&d,&e),c+=a;
    for(int i=0;i<=e;i++) ans=(ans+1ll*C(b,i)*C(d,e-i)%p*A(a,i)%p*A(c-i,e-i)%p)%p;
    printf("%d\n",ans);
}
```

#### [P3166 数三角形](https://www.luogu.com.cn/problem/P3166)

正难则反
我们发现水平的非常容易排除
但有一种抽象的斜线很费劲
那么怎么办？

我们枚举每一个点
然后把他和原点作为线段起终点
在上面找点的数量是容易的
然后乘二计算出负斜率的情况即可
最后把它移位即可

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
inline ll f(ll x){return x*(x-1)*(x-2)/6;}
ll gcd(ll a,ll b){return (b==0)?a:gcd(b,a%b);}
int main(){
	ll n,m,ans; scanf("%lld%lld",&n,&m),ans=f((n+1)*(m+1))-(n+1)*f(m+1)-(m+1)*f(n+1);
	for(ll i=1;i<=n;i++) for(ll j=1;j<=m;j++) ans-=(n-i+1)*(m-j+1)*(gcd(i,j)-1)*2;
	printf("%lld\n",ans);
}
```

#### [LibreOJ 10235. 序列统计](https://loj.ac/p/10235)

如果这个控制长度为定值那他就是插板法模板
可然而他并不是
我们设 $r-l=\Delta$
则长为 $i$ 个的方案隔板不难得到 $C_{\Delta+i}^i$

接下来我们就需要一个性质
我们的答案实际上是 $\sum_{i=1}^n C_{\Delta+i}^i$
先放结论：$\sum_{i=1}^{n}C_{m+i}^i=C_{m+n+1}^n-1$
这个我们可以叫 **+1-1定理**
或者这个用小球可空分组形式理解
就是 $m$ 个小球分 $2\to n+1$ 组的方案
等于 $m+1$ 个小球分成 $n+1$ 组的方案减去一
我们简单把他变一下：$m$ 个小球分成 $1\to n$ 个组的方案，等于 $m+1$ 个小球分 $n$ 组的方案

有了这个我们有可以开干了
但是这个为什么呢？
$$
\begin{aligned}
\sum_{i=1}^n C_{m+i}^i
&=C_{m+1}^{0}-1+\sum_{i=1}^n C_{m+i}^{i}\\
&=C_{m+1}^{0}+C_{m+1}^{1}-1+\sum_{i=2}^n C_{m+i}^{i}\\
&=C_{m+2}^{1}-1+\sum_{i=2}^n C_{m+i}^{i}\\
&=C_{m+2}^{1}+C_{m+2}^{2}-1+\sum_{i=3}^n C_{m+i}^{i}\\
&=C_{m+3}^{2}-1+\sum_{i=3}^n C_{m+i}^{i}\\
&=\dots\\
&=C_{m+k}^{k-1}-1+\sum_{i=k}^n C_{m+i}^{i}\\
\texttt{let k=n}\\
&=C_{m+n}^{n-1}-1+C_{m+n}^{n}\\
&=C_{m+n+1}^{n}-1\\
\end{aligned}
$$
~~好了好了，不唐了~~
~~还得打 Lucas，唉。。。~~

```cpp
#include<bits/stdc++.h>
using namespace std;
const int p=1000003;
int inv[p+2],f[p+2];
int C(int n,int m){return (n<m)?0:1ll*f[n]*inv[f[n-m]]%p*inv[f[m]]%p;}
int Lucas(int n,int m){return (n<p)?C(n,m):1ll*Lucas(n/p,m/p)*C(n%p,m%p)%p;}
int main(){
    inv[1]=f[0]=f[1]=1; for(int i=2;i<p;i++) f[i]=1ll*f[i-1]*i%p,inv[i]=(-1ll*p/i*inv[p%i]%p+p)%p;
    int n,l,r,t; scanf("%d",&t);
    while(scanf("%d%d%d",&n,&l,&r)!=EOF) printf("%d\n",(Lucas(r-l+1+n,n)-1+p)%p);
}
```

#### [P4345 超能粒子炮](https://www.luogu.com.cn/problem/P4345)

求 $\sum_{i=0}^{k} C_{n}^{i} \bmod 2333$

这题对于只会背板子的蒟蒻作者来说只能硬算了
我们看到提示用 Lucas 推式子试一下？
Lucas 这个东西本质上是让我们来对原来的 $m,n$ 拆成一个 $p$ 进制的数字
而我们发现他最多只有六位
那我们显然可以把原式子变成 $\sum_{\overline{j_1j_2j_3j_4j_5j_6}=0}^\overline{i_1i_2i_3i_4i_5i_6} C_{a_1}^{j_1}C_{a_2}^{j_2}C_{a_3}^{j_3}C_{a_4}^{j_4}C_{a_5}^{j_5}C_{a_6}^{j_6}$
怎么办？
数位 DP！
我们设 $dp_{i,j}$ 表示一直到第 $i$ 位前均为零而首位为 $j$ 的方案
那么显然有 $dp_{i,j}=C^j_{a_i}\sum_{j=0}^{2332}dp_{i+1,j}$
然后怎么搞怎么搞。。。

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int p=2333;
int dp[8][p+10]={},inv[p+10],f[p+10];
inline int C(int n,int m){if(m>n) return 0; return f[n]*inv[f[n-m]]%p*inv[f[m]]%p;}
template <class T> inline void read(T &x){
	char k=getchar(); x=0;
	while(!isdigit(k)) k=getchar();
	while(isdigit(k)) x=x*10+k-'0',k=getchar();
}
int main(){
	int T,len,ans,N[8],K[8]; ll n,k; read(T),inv[1]=f[0]=f[1]=1;
	for(int i=2;i<p;i++) f[i]=f[i-1]*i%p,inv[i]=(-p/i*inv[p%i]%p+p)%p;
	while(T--){
		read(n),read(k),len=ans=0,++k;
		for(int i=1;i<=6;i++) N[i]=n%p,K[i]=k%p,n/=p,k/=p,len=(K[i])?i:len;
		for(int i=0;i<p;i++) dp[1][i]=C(N[1],i);
		for(int i=2,tmp;i<=len;i++){
			tmp=0; for(int j=0;j<p;j++) tmp=(tmp+dp[i-1][j])%p;
			for(int j=0;j<p;j++) dp[i][j]=C(N[i],j)*tmp%p;
		}
		for(int i=0;i<K[len];i++) ans=(ans+dp[len][i])%p;
		for(int lock=len,tmp=1;lock>1;lock--){
			tmp=tmp*C(N[lock],K[lock])%p;
			for(int i=0;i<K[lock-1];i++) ans=(ans+tmp*dp[lock-1][i])%p;
		}
		printf("%d\n",ans);
	}
	return 0;
}
```

一通乱搞，TLE 70 pts
我们计算一下时间复杂度 $O(6Tp)=O(1e9+)$
我们主要是需要加速求 $dp$ 数组的过程
仔细思考发现并没有必要把所有的 dp 之都显式的搞出来
我们真正需要的是每一层的和
需要每个求出来么？不能够啊！
我们每一层的和 $sum_i=sum_{i-1}*\sum_{i=1}^{N_i} C_{N_i}^i$
我们就可以直接预处理所有的阶乘及其前缀和
然后用的时候 `inline int dp(int i,int j){return sum[i-1]*C(N[i],j);}`

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int p=2333;
int C[p+5][p+5],CC[p+5][p+5],len,ans,N[8],K[8],sum[8],T;
inline int dp(int i,int j){return sum[i-1]*C[N[i]][j]%p;}
int main(){
	ll n,k; scanf("%d",&T),C[0][0]=sum[0]=1;
    for(int i=0;i<p;i++) for(int j=0;j<=i;j++) C[i+1][j]=(C[i+1][j]+C[i][j])%p,C[i+1][j+1]=(C[i+1][j+1]+C[i][j])%p;
    for(int i=0;i<p;i++) CC[i][0]=C[i][0];
    for(int i=1;i<p;i++) for(int j=1;j<=i;j++) CC[i][j]=(C[i][j]+CC[i][j-1])%p;
	while(T--){
		scanf("%lld%lld",&n,&k),len=ans=0,++k;
		for(int i=1;i<=6;i++) N[i]=n%p,K[i]=k%p,n/=p,k/=p,len=(K[i])?i:len;
		for(int i=1;i<=6;i++) sum[i]=sum[i-1]*CC[N[i]][N[i]]%p;
		for(int i=0;i<K[len];i++) ans=(ans+dp(len,i))%p;
		for(int lock=len,tmp=1;lock>1;lock--){
			tmp=tmp*C[N[lock]][K[lock]]%p;
			for(int i=0;i<K[lock-1];i++) ans=(ans+tmp*dp(lock-1,i))%p; //TLE
		}
		printf("%d\n",ans);
	}
	return 0;
}
```

我们发现还是 TLE
由于有一个双循环语句又把我们卡成了无效优化
但我们发现没这一句也可以优化掉
$\sum_{i=0}^{K[lock-1]-1}dp_{lock-1,i}=sum_{lock-2}\sum_{i=0}^{K[lock-1]-1}C_{N_{lock-1}}^i=sum_{lock-2}CC^{K[lock-1]-1}_{N_{lock-1}}$

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int p=2333;
int C[p+5][p+5],CC[p+5][p+5],len,ans,N[8],K[8],sum[8],T;
inline int dp(int i,int j){return sum[i-1]*C[N[i]][j]%p;}
int main(){
	ll n,k; scanf("%d",&T),C[0][0]=sum[0]=1;
    for(int i=0;i<p;i++) for(int j=0;j<=i;j++) C[i+1][j]=(C[i+1][j]+C[i][j])%p,C[i+1][j+1]=(C[i+1][j+1]+C[i][j])%p;
    for(int i=0;i<p;i++) CC[i][0]=C[i][0];
    for(int i=0;i<p;i++) for(int j=1;j<p;j++) CC[i][j]=(C[i][j]+CC[i][j-1])%p;
	while(T--){
		scanf("%lld%lld",&n,&k),len=ans=0,++k;
		for(int i=1;i<=6;i++) N[i]=n%p,K[i]=k%p,n/=p,k/=p,len=(K[i])?i:len;
		for(int i=1;i<=6;i++) sum[i]=sum[i-1]*CC[N[i]][N[i]]%p;
		for(int i=0;i<K[len];i++) ans=(ans+dp(len,i))%p;
		for(int lock=len,tmp=1;lock>1;lock--){
			tmp=tmp*C[N[lock]][K[lock]]%p;
			if(K[lock-1]>0) ans=(ans+1ll*tmp*sum[lock-2]%p*CC[N[lock-1]][K[lock-1]-1]%p)%p; 
		}
		printf("%d\n",ans);
	}
	return 0;
}
```

#### [P2183 礼物](https://www.luogu.com.cn/problem/P2183)

思考先求出从 $n$ 个礼物里面取 $\sum w_i$ 个的排列
然后除掉每一个人的个数的全排列
答案即为 $A_n^{\sum w_i}/\prod (w_i!)=\frac{n!}{(n-tot)!\prod (w_i!)}$

However
Lucas 只能求组合数
这就要求我们把它变成一个组合数
我们推一波式子
$$
\begin{aligned}
&\frac{n!}{(n-tot)!\prod_{i=1}^k (w_i!)}\\
=&\frac{n!}{(n-tot)!w_1!\prod_{i=2}^k (w_i!)}*\frac{(n-tot+w_1)!}{(n-tot+w_1)!}\\
=&\frac{(n-tot+w_1)!}{(n-tot)!w_1!}*\frac{n!}{(n-tot+w_1)!\prod_{i=2}^k (w_i!)}\\
=&C_{n-tot+w_1}^{w_1}*\frac{n!}{(n-tot+w_1)!\prod_{i=2}^k (w_i!)}\\
=&C_{n-tot+w_1}^{w_1}C_{n-tot+w_1+w_2}^{w_2}*\frac{n!}{(n-tot+w_1+w_2)!\prod_{i=3}^k (w_i!)}\\
=&\dots\\
=&C_{n-tot+w_1}^{w_1}C_{n-tot+w_1+w_2}^{w_2}\dots C_{n-tot+w_1+w_2+\dots+w_{k-1}}^{w_{k-1}}*\frac{n!}{(n-tot+w_1+w_2+\dots+w_{k-1})!w_k!}\\
=&C_{n-tot+w_1}^{w_1}C_{n-tot+w_1+w_2}^{w_2}\dots C_{n-tot+w_1+w_2+\dots+w_{k-1}}^{w_{k-1}}C_{n}^{w_k}\\
=&\prod_{i=1}^kC_{n-tot+\sum_{j=1}^iw_j}^{w_i}
\end{aligned}
$$
然后我们使用 exLucas 求解即可

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll; 
inline ll G(ll n,ll P){ll ans=0; for(ll i=P;i<=n;i*=P) ans+=n/i; return ans;}
inline ll Pow(ll n,ll m,ll p){ll ans=1; while(m){if(m&1) ans=ans*n%p; n=n*n%p,m>>=1;} return ans;}
void Exgcd(ll a,ll b,ll &x,ll &y){if(b==0) x=1,y=0; else Exgcd(b,a%b,y,x),y-=a/b*x;}
ll inv(ll x,ll p){ll m,n; Exgcd(x%p,p,m,n); return (m%p+p)%p;}
ll F(ll n,ll p,ll P){
	if(n==0) return 1; ll ans=1;
	for(ll i=1;i<=P;i++) if(i%p!=0) ans=ans*i%P;
	ans=Pow(ans,n/P,P)*F(n/p,p,P)%P;
	for(ll i=n/P*P;i<=n;i++) if(i%p!=0) ans=ans*i%P;
	return ans;
}
ll a[40],b[40],ans[40]; int cnt;
ll Lucas(ll n,ll m,ll p){
	ll tmp=p,answer=0; cnt=0;
	for(ll i=2;i*i<=p;i++){
		if(p%i) continue;
		a[++cnt]=i,b[cnt]=1;
		while(p%i==0) p/=i,b[cnt]*=i;
	}
	if(p>1) a[++cnt]=p,b[cnt]=p;
	for(int i=1;i<=cnt;i++) ans[i]=F(n,a[i],b[i])*inv(F(n-m,a[i],b[i]),b[i])%b[i]*inv(F(m,a[i],b[i]),b[i])%b[i]*Pow(a[i],G(n,a[i])-G(m,a[i])-G(n-m,a[i]),b[i])%b[i];
	for(int i=1;i<=cnt;i++) answer=(answer+tmp/b[i]*inv(tmp/b[i],b[i])%tmp*ans[i]%tmp)%tmp;
	return answer; 
}
int main(){
	int P,n,m,w[10],tot=0; ll ans=1; scanf("%d%d%d",&P,&n,&m); 
	for(int i=1;i<=m;i++) scanf("%d",&w[i]),tot+=w[i]; if(tot>n) puts("Impossible"),exit(0); else n-=tot;
	for(int i=1;i<=m;i++) n+=w[i],ans=ans*Lucas(n,w[i],P)%P;
	printf("%lld\n",ans);
	return 0;
}
```

式子还可以简单一些
我们发现
从 $n$ 个里面选 $w_1$ 个，就有 $C_n^{w_1}$ 种方案；
从 $n-w_1$ 个里面选 $w_2$ 个，就有 $C_{n-w_1}^{w_2}$ 种方案；
以此类推
答案即为 $\prod_{i=1}^k C^{w_i}_{n-\sum_{j=1}^{i-1}w_j}$
#### [LibreOJ 10238. 网格](https://loj.ac/p/10238)
学了 exCatalan 显然模板（这题过于板了
~~火车头卡 Ac 喜提最慢代码~~

```cpp
#include<bits/stdc++.h>
#define A a.k
#define B b.k
#define C c.k
#define debug putchar('\n')
using namespace std;
class Big{
	public:
		template<class T> void scan(T x){k.clear(); while(x) k.push_back(x%10),x/=10;}
		void print(){if(k.empty()) putchar('0'); else for(int i=k.size()-1;i>=0;i--){putchar(k[i]+'0'); if(!isdigit(k[i]+'0')) puts("Error!"),exit(0);} debug;}
		friend Big operator *(Big a,int b){
			Big c; int la=A.size();
			A.resize(la+5);
			// for(int i=1;i<=10;i++) A.push_back(0);
			for(int i=0;i<la;i++) A[i]*=b;
			for(int i=0;i<=la+6;i++) A[i+1]+=A[i]/10,A[i]%=10;
			while(A.back()==0) A.pop_back(); return a;
		} 
		friend Big operator /(Big a,int b){
			int la=A.size(); unsigned long long tmp=0;
			for(int i=la-1;i>=0;i--){
				tmp=tmp*10+A[i];
				A[i]=tmp/b,tmp%=b;
			}
			while(A.back()==0) A.pop_back(); return a;
		}
	private:
		vector<unsigned int> k;
}ans;
int main(){
	int n,m; scanf("%d%d",&m,&n),ans.scan(m-n+1);
	for(int i=m+2;i<=m+n;i++) ans=ans*i;//,ans.print();
	for(int i=2;i<=n;i++) ans=ans/i;
	ans.print();
	return 0;
}
```

#### [LibreOJ 10239. 有趣的数列](https://loj.ac/p/10239)

我们考虑定义“左加”为在奇数项放下一个，“右加”同理
发现就是裸 Catalan
然而看了眼数据。。。
MD 没说 P 是质数，这死玩应不会还得 exLucas 吧。。。
要不还是老老实实用定义式 $C_{2n}^n-C_{2n}^{n+1}$ 吧。。。
不过就是码亿点点吗。。。

> exLucas 限时返厂
> 老定义 $P=p^k,\Delta=\lfloor\frac np\rfloor,\Delta'=\lfloor\frac nP\rfloor$
> $$
> \begin{aligned}
> C_n^m\bmod P&=\frac{F(n)}{F(m)F(n-m)}*p^{G(n)-G(m)-G(n-m)}\\
> n!&\equiv p^{\Delta}\Delta!\prod_{p\not\mid i}^n i\\
> F(n)&=F(\Delta)\prod_{p\not\mid i}^n i=F(\Delta)(\prod_{p\not\mid i}^{P} i)^{\Delta'}\prod_{p\not\mid i}^{P\Delta'}i\\
> G(n)&=\sum_{i=1}^{p^i\le n} \lfloor\frac n{p^i}\rfloor
> \end{aligned}
> $$
> 

![image](https://imgs.qiubiaoqing.com/qiubiaoqing/imgs/607dfd6e30efcRUD.gif)

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll Pow(ll a,int b,ll p){ll ans=1; while(b){if(b&1) ans=ans*a%p; b>>=1,a=a*a%p;} return ans;}
void exGcd(ll a,ll b,ll &x,ll &y){if(b==0) x=1,y=0; else exGcd(b,a%b,y,x),y-=a/b*x;}
ll Inv(ll n,ll p){ll x,y; exGcd(n,p,x,y); return (x%p+p)%p;}
int G(ll n,ll p){int tmp=0; for(ll i=p;i<=n;i*=p) tmp+=n/i; return tmp;}
ll F(ll n,ll p,ll P){
	if(n==0||n==1) return 1;
	ll ans=1;
	for(ll i=1;i<=n/P*P;i++) if(i%p!=0) ans=ans*i%P;
	ans=Pow(ans,n/P,P)*F(n/p,p,P)%P;
	for(ll i=n/P*P+1;i<=n;i++) if(i%p!=0) ans=ans*i%P;
	return ans;
}
ll CRT(ll *_ans,ll *_mod,ll P,int cnt){ll ans=0; for(int i=1;i<=cnt;i++) ans=(ans+P/_mod[i]*Inv(P/_mod[i],_mod[i])%P*_ans[i]%P)%P; return ans;}
ll ans[101],mod[101],tmp[101];
int cnt;
ll exLucas(ll n,ll m,ll p){
	cnt=0; ll _p=p;
	for(int i=2;i*i<=_p;i++) if(_p%i==0){tmp[++cnt]=i,mod[cnt]=1; while(_p%i==0) mod[cnt]*=i,_p/=i;}
	if(_p>1) tmp[++cnt]=_p,mod[cnt]=_p;
	for(int i=1;i<=cnt;i++) ans[i]=F(n,tmp[i],mod[i])*Inv(F(m,tmp[i],mod[i]),mod[i])%mod[i]*Inv(F(n-m,tmp[i],mod[i]),mod[i])%mod[i]*Pow(tmp[i],G(n,tmp[i])-G(m,tmp[i])-G(n-m,tmp[i]),mod[i])%mod[i];
	return CRT(ans,mod,p,cnt);
}
int main(){
	ll n,p; scanf("%lld%lld",&n,&p),printf("%lld\n",(exLucas(2*n,n,p)-exLucas(2*n,n+1,p)+p)%p);
	return 0;
}
```

#### [P2532 树屋阶梯](https://www.luogu.com.cn/problem/P2532)

前三个熟悉的 125 猜测可能是 Catalan
推导：[Link](https://www.luogu.com.cn/article/joifxlnw)

```cpp
#include<bits/stdc++.h>
#define A a.k
#define B b.k
#define C c.k
#define debug putchar('\n')
using namespace std;
class Big{
	public:
		template<class T> void scan(T x){k.clear(); while(x) k.push_back(x%10),x/=10;}
		void print(){if(k.empty()) putchar('0'); else for(int i=k.size()-1;i>=0;i--){putchar(k[i]+'0'); if(!isdigit(k[i]+'0')) puts("Error!"),exit(0);} debug;}
		friend Big operator *(Big a,int b){
			Big c; int la=A.size();
			// A.resize(la+5);
			for(int i=1;i<=10;i++) A.push_back(0);
			for(int i=0;i<la;i++) A[i]*=b;
			for(int i=0;i<=la+6;i++) A[i+1]+=A[i]/10,A[i]%=10;
			while(A.back()==0) A.pop_back(); return a;
		} 
		friend Big operator /(Big a,int b){
			int la=A.size(); unsigned long long tmp=0;
			for(int i=la-1;i>=0;i--){
				tmp=tmp*10+A[i];
				A[i]=tmp/b,tmp%=b;
			}
			while(A.back()==0) A.pop_back(); return a;
		}
	private:
		vector<unsigned int> k;
}ans;
int main(){
	int n; scanf("%d",&n),ans.scan(1);
    for(int i=n+2;i<=n+n;i++) ans=ans*i;
    for(int i=2;i<=n;i++) ans=ans/i;
    ans.print();
	return 0;
}
```

