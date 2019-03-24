# python 笔记

### 18.03.2019

今天安装上了python和opencv，

检查opencv的时候，一直都显示没有安装上。

GitHub上有一个回答，还是不错的。

![1552900115999](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1552900115999.png)

终于是把opencv安装上了。



##### 1. Numbers 

- 用__来表示上面已经计算完的结果。

##### 2. String

- 用单引号来分解比较长的string

````python
>>> text = ('Put several strings within parentheses '
...         'to have them joined together.')
>>> text
'Put several strings within parentheses to have them joined together.'
````

- len返回string的长度。
- Stirng[:2]表示从第二个字符往前(也就是1，2两个字符**保留**)，String[1:]表示从第一个字符往后（就是第一个字符保留）

##### 3. Lists

- 数组可以通过 + 添加数组的长度。只是添加到后面，不会自动排序。

- letters数组里从第二个元素开始，删除3个元素。

  ````python
  letters[2:5] = []
  
  ['a', 'b', 'f', 'g']
  ````

- python里面的连续逗号：

  ````python
  >>> a, b = 0, 1
  >>> while a < 1000:
  ...     print(a, end=',')
  ...     a, b = b, a+b
  ````

  最后一个的意思是 a = b , b = a + b

#### Control Flow Tools

##### 1. pass 

​	pass就是在函数里定义一个空。目的是什么也不做。可以当成一个占位符，用来先让程序能运行起来。find 



#### 高级函数

##### 1. 函数就是一个一个变量。如果一个函数被赋值成了另一个变量，需要重启python才能用。

````python
>>> abs(-10)
10

>>> abs
<built-in function abs>

>>> f = abs
>>> f
<built-in function abs>
````

##### 2. 传入函数

变量可以指向函数，函数的参数可以接受变量。那么，一个函数就可以接受另一个函数作为参数。这个就叫高阶函数。

###### map函数：

作为一个高阶函数，可以直接将一个函数作用到传入的数组上。

````python
 def f(x):
    return x * x

 r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
 list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
````

map/reduce 练习：

````
def normalize(name):
    name = name[0].upper()+name[1:].lower()
    return name


L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)
````



这个函数就是将f的平方函数，直接作用在数组上，并输出。简单明了的看出，是进行了什么运算。

###### reduce 函数

```python
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
```

###### filter 函数

filter函数一定要在函数中定义，返回值的true 和 false

````python
def is_palindrome(n):
    nn = str(n)
    return nn == nn[::-1] 

print (list(filter(is_palindrome,range(1,1000))))
````

###### sorten函数

可以直接按**数字大小**排序，也可以按照**绝对值大小**排序。



































