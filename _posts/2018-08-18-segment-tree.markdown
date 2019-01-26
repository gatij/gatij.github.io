---
layout:     post
title:      "Segment Tree : A mysterious weapon of Comptetive Programmers"
subtitle:   "Advance Data Structure:Simple C++ Implementation"
date:       2018-08-18 12:40:00
author:     "Gatij Jain"
header-img: "img/post-ST-18-8-18.JPG"
header-mask: 0.3
catalog:    true
tags:
    - Data Structure
    - C++
    - CP
    - Algorithms
---

**Segment Tree** is one of the most used advanced data structure in comptetive programming. In this post we will explore the situations where we can use this data structure and we will implement it using C++ from scratch.

As the name suggests this data structure is applicable for segment i.e Range Query  problems. Segment Tree is basically an another way to arrange data in [**Binary Tree**](https://www.wikiwand.com/en/Binary_tree). This data structure can efficiently answer **Dynamic Range Queries**.

### **Dynamic Range Queries**
The problems in which the given data is frequently updated and queried (usually on range) in a range can be categorized in **Dynamic Range problems**.And for these problems the **Pre-Procesing techniques** like PreOrder Array becomes useless.
One of the best popular basic problem of such type is to find minimum in Range [i..j].This is called **Range Minimum Query(RMQ)** problem.The same concept can be extended to **Range Maximum Query** and **Range Sum Query**.

### Learn by example
Lets consider Array A=[2,1,7,5,4,6,4,3,5,4,9].We need to find the sum of the segment  [L,R] for each query.Where 1<=L,R<=11.(Array is 0-indexed)

### Approach 1
This is naive approach ,we iterate through each element in range[L,R] and store thier sum in a variable and answer for that query is sum stored in that variable.
But this apprach is good only is there are very few queries.Because complexity of finding sum of range using this approach is O(N).
And if there are large number of queries then use of this technique is terrible.

### Approach 2
In this approach we make a PreOrder array S such that S[i]=S[i-1]+A[i] if i>0 and  
S[0]=A[0].Now the S=[2,3,10,15,19,25,29,32,37,41,50].Now for each query on segment 
[L,R] the sum of all elements of segment is (S[R-1]-S[L-2]).
Ex: if [L,R]=[3,5] then answer of query=(S[5-1]-S[3-2])=16.
That's great as we can answer each query in O(1) time only.
But let's modify our problem simply what if we want to know the minimum/maximum number of the segment [L,R].
And here our PreOrder array will become useless.Therefore we need some new technique to deal with this.And that is **"Segment Tree"**.

### Approach 3 - Segment Tree
Before making our hand dirty with code and implementation let's try to understand the concept by figures and examples.
consider array A=[2,1,7,5,4,6,4,3,5,4,9] now we try to find how the segment tree
of this array looks like. Here I take the example of segment tree for range minimum query. 
![](https://drive.google.com/file/d/118Of4qd3z2BdrKrRKW11JCf89ZRVzzHd/view?usp=sharing)
![jpg](/img/in-post/post-segment-tree/seg_tree_1.jpg)

We store binary tree structure as follows:
if root is at index "**p**" and left child of that root is at "**2p**" and right child at "**2p+1**".

So in figure above the index mention in **blue color** are the indices of segment tree. Values written in nodes (cicle) are the values of the **index**(not the value itself) from original array A.And red color segment under each node represents the range of that node.

Let's try to undestand how this can be formed and how it can solve our problem.
First we just forget the blue color indices as they are the part of implementation and storing of segment tree.We need to understand how this tree structure formed.
Start from bottom of the tree all leaf nodes are nothing but the nodes representing array A **index** value.Actually we are storing the **index of the minimum element**of array A in tree node representing the range written in **[L,R]** **(red color)**.
As leaf node have range of single element only hence the minimum element's index of that range is the index of that element itself.Therefore every leaf node have the index of the element of array A of that range.
Example: [0,0] = 0 (node with 16 in blue color)
         [1,1] = 1 (node with 17 in blue color)
         and so on...

After that let's try to find what the node representing range[0,1] store.
Yes exactly what you are thinking it should store the index "0" or index "1" and that depends on what is minimum A[0] or A[1].So in our example A[0]=2 and A[1]=1
as A[1] < A[0] so we need to store "1" in node of segment tree.And we need to repeat same process for every node having left child as well as right child.
By this process:
               [0,1] = 1 (node with 8 in blue color)
               [3,4] = 4 (node with 10 in blue color)
               ....
               [6,10] = 7 (node with 3 in blue color)
               ....
               [0,10] = 1 (node with 1 in blue color)

Now we have good idea of what the node of segment tree is representing and what is the logic for selcting that value at node.
I repeat once more every node of segment tree is representing the index of the minimum element of the array A of that range.And we select that index by checking which of the two child of that node have the index of minimum element.

So our concept of segment-tree is clear.By the way you can also store the minimum value itself from array A on the place of index of minimum element,but here in example we store the index.

Now our next big task is how to build that segement tree with the help of code,because that's the way we can use it in contest problem.
So lets start the game.

### Implementation 

We build segment tree as **recursive** approach.
First a basic c++ Class body is given:(Don't get panic I will explain every bit of code ).

```c++
#include<bits/stdc++.h>
using namepsace std;

class SegmentTree{
	int n;//size of the array on which segment tree is build i.e A
	vector<int>A;//Array for segment tree is build
	vector<int>st;//array that store the segment tree 
	
	int left(int p)//finding left child of index p
	{
		return (p<<1);//2*p
	}
	
	int right(int p)//finding right child of index p
	{
		return (p<<1)+1;//2*p + 1
	}
	
	//This function build the segment tree i.e. fill the value in array "st"
	void build(int p,int l,int r)
	{
		if(l==r)
	      st[p]=l;//or st[p]=r
	    else
	    {
	    	build(left(p),l,(l+r)/2);//build the segment tree for left child first
	    	build(right(p),((l+r)/2)+1,r);//build the segment tree for right child
			//li is index of minimum element of range[l,(l+r)/2]
			int li=st[left(p)];
	    	//ri is index of minimum element of range[((l+r)/2)+1,r]
	    	int ri=st[right(p)];
	    	//taking decision that minimum element index store in parent node
	    	if(A[li]<A[ri])
	    	{
	    		st[p]=li;
			}
			else
			{
				st[p]=ri;
			}
		}
	}

public:
	//constructor for initialization of segment tree
	SegmentTree(const vector<int> &_A)
	{
		A=_A;
		n=(int)A.size();
		st.assign(4*n,0);
		build(1,0,n-1);
	}	
	
};
```

Let's try to decode the **build()** function.In our example we will build our tree as by calling **"build(1,0,10)"**(as size n=11).Calling of recursive function looks like:
```
                            build(1,0,10)
                            |          | 
                  build(2,0,5)        build(3,6,10)
                  |  |                |          |
       build(4,0,2)  build(5,3,5)  build(6,6,8) build(7,9,10)     
                                  .
                                  .
                                  .

```                                  

this recursive calling goes on until some call like **build(p,x,y)** make where x=y.
So these call will end by filling leaf nodes of segment tree by definition in 
**if** block of **build()** in above code.  ( **where l==r**)

That's how the segment tree build up.

Yes we have learnt how to build but what we actually need is how to use this.
okay so ready to learn again how to get query answer by the segment tree builded already.

```c++
//Range_Minimum_Query function for range interval [i,j]
int rmq(int p,int l,int r,int i,int j)
{
	if(i>r || j<l)//current segment is outside the query range
		return -1;
	if(l>=i && r<=j)//current segment is inside the query range
		return st[p];
	//compute the min element index in left and right part of the interval
	int li=rmq(left(p),l,(l+r)/2,i,j);
	int ri=rmq(right(p),l,(l+r)/2,i,j);

    //if we try to access segment outside query
	if(li==-1) 
		return ri;
	if(ri==-1)
		return li;
    //similar as build() function
	if(A[li]<A[ri])
		return li;
	else
		return ri;

}

```

This is also recursive implementation and when you draw the recursion tree for this you will get the logic why it is working lets take one example:


Lets first get rid of redundant code for each query as first three arguments of the **rmq()** function are always same for each query.We are using the concept of **function overloading** here.

```c++
//in public scope of class SegmentTree 
int rmq(int i,int j)
{
	return rmq(1,0,n-1,i,j);//rmq defined in private scope of class.
} 

```

Ex1- suppose query is [L,R] = [1,8] i.e (index 0 to 7 in actual array A).So answer should be min of array [2,1,7,5,4,6,4,3] and it is "1" but **rmq(1,0,10,0,7)** should return index of min element that is also "1" in this case.

```
                             rmq(1,0,10,0,7)//return 1 as A[1]<A[7]
                             |            |
               rmq(2,0,5,0,7)//return 1    rmq(3,6,10,0,7)//return 7
                                          |             |
                          rmq(6,6,8,0,7)//return 7   rmq(7,9,10,0,7)//return -1
                           |            |                           
                   rmq(12,6,7,0,7) rmq(13,8,8,0,7)
              // return st[12]=7             //return -1   
```
In the recursion tree where the rmq function returns -1 is the case in which range of [l,r] is outside [0,7] and hence we ignore that section by returning the the value of other child to the parent .
And when [l,r] is completely inside [0,7] then it return the st[p] i.e index of min element of that [l,r]. 

### Code for Further Use
This is code snippet of Range Minimum Query Segment Tree also uploaded on my github [Gist](https://gist.github.com/gatij/035276170724aeac8587057513224b87).

<script src="https://gist.github.com/gatij/035276170724aeac8587057513224b87.js"></script>

             



