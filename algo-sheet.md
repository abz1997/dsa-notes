From: https://techinterviewhandbook.org

# Array

Note that because both arrays and strings are sequences (a string is an array of characters), most of the techniques here will apply to string problems.

### Sliding window
Master the sliding window technique that applies to many subarray/substring problems. In a sliding window, the two pointers usually move in the same direction will never overtake each other. This ensures that each value is only visited at most twice and the time complexity is still O(n). Examples: Longest Substring Without Repeating Characters, Minimum Size Subarray Sum, Minimum Window Substring

Template
```
1. Initialize two pointers, start and end, both at the 0th index.

2. Initialize any needed variables. For example:
   - max_sum for storing the maximum sum of subarray
   - current_sum for storing the sum of the current window
   - any other specific variables you need

3. Iterate over the array/string using the end pointer:

   while end < length_of_array:
       
       a. Add the current element to current_sum (or perform some operation)

       b. While the current window meets a certain condition (like current_sum exceeds a value, the window has more than a specific number of characters, etc.):
          i. Possibly update max_sum or any other variables
          ii. Remove the element at start pointer from current_sum (or perform the opposite operation)
          iii. Increment the start pointer to make the window smaller

       c. Increment the end pointer

4. After the loop, the answer could be in max_sum or any other variables you used.
```

Example
```python
# Find the maximum sum subarray of size `k`:
def maxSumSubarray(arr: List, k: int): 

    start = 0
    currentSum = 0
    maxSum = float(-inf)

    for end in range(len(arr)):
        currentSum += arr[end]

        if end - start + 1 == k:
            maxSum = max(maxSum, currentSum)
            currentSum -= arr[start]
            start += 1

    return maxSum
```

### Two pointers
Two pointers is a more general version of sliding window where the pointers can cross each other and can be on different arrays. Examples: Sort Colors, Palindromic Substrings

When you are given two arrays to process, it is common to have one index per array (pointer) to traverse/compare the both of them, incrementing one of the pointers when relevant. For example, we use this approach to merge two sorted arrays. Examples: Merge Sorted Array

Template
```
1. Initialize two pointers. Depending on the problem:
   - They can start at the beginning and end of the array (`start = 0, end = length_of_array - 1`), which is typical for sorted arrays.
   - Or they can start both at the beginning (`left = 0, right = 1`) or any other positions based on the requirement.

2. Use a loop (typically a while loop) to iterate while the pointers meet the criteria for traversal, e.g., `start < end`.

3. Inside the loop:
   a. Check the current elements at the two pointers.
   b. Based on the problem, decide how to adjust the pointers. Common decisions are:
      i. If the current elements meet some condition, process the current elements and then adjust the pointers (either moving `start` forward or `end` backward or both).
      ii. If the current elements don't meet the condition, adjust one of the pointers (or both) without processing the elements.

4. Continue until the loop ends.

5. The solution might be obtained during the loop's iterations, or after the loop based on the processed elements.
```

Example
```python
# In a sorted array, find a pair of elements that sum up to a given target:
def twoSumSorted(arr: List, target: int):

    start = 0
    end = len(arr) - 1

    while start < end:
        currentSum = arr[start] + arr[end]

        if currentSum == target:
            return arr[start], arr[end]
        elif currentSum < target:
            start += 1
        else:
            end -= 1

    return None # No pair found
```

### Traversing from the right
Sometimes you can traverse the array starting from the right instead of the conventional approach of from the left. Examples: Daily Temperatures, Number of Visible People in a Queue

### Sorting the array
Is the array sorted or partially sorted? If it is, some form of binary search should be possible. This also usually means that the interviewer is looking for a solution that is faster than O(n).

Can you sort the array? Sometimes sorting the array first may significantly simplify the problem. Obviously this would not work if the order of array elements need to be preserved. Examples: Merge Intervals, Non-overlapping Intervals

### Precomputation
For questions where summation or multiplication of a subarray is involved, pre-computation using hashing or a prefix/suffix sum/product might be useful. Examples: Product of Array Except Self, Minimum Size Subarray Sum, LeetCode questions tagged "prefix-sum".

matrix prefix sum
- we want to calculate all elements in matrix before. So lets say we want to calculate a prefix sum for a particular cell. That means we assume that is bottom right of rectangle and the sum is all the points in that.
- The prefix is the sum of all points up to that point (the rows and columns before it)
```python
A = [
    [1, 2, 1],
    [3, 4, 1],
    [2, 1, 5]
]


# Assume n = number of rows, m = number of columns
prefix = [[0] * m for _ in range(n)]

for i in range(n):
    for j in range(m):
        prefix[i][j] = A[i][j]
        # A is the original matrix
        if i > 0:
            prefix[i][j] += prefix[i-1][j]
        if j > 0:
            prefix[i][j] += prefix[i][j-1]
        if i > 0 and j > 0:
            prefix[i][j] -= prefix[i-1][j-1]

# Prefix then becomes
prefix = [
    [ 1,  3,  4 ],
    [ 4, 10, 12 ],
    [ 6, 13, 20 ]
]


```

To query the sum of any submatrix where r1,c1 and r2,c2 are the top left and bottom right of rectangle:
The equation is:
```
sum = prefix[r2][c2] − prefix[r2][c1−1] − prefix[r1−1][c2] + prefix[r1−1][c1−1]
```

```python
# Query shape with top-left = (1,1) and bottom-right = (2,2)
# [4, 1]
# [1, 5]
# sum=prefix[2][2]−prefix[2][0]−prefix[0][2]+prefix[0][0]


def query_submatrix_sum(prefix, r1, c1, r2, c2):
    result = prefix[r2][c2]

    # all the columns before
    if c1 > 0:
        result -= prefix[r2][c1 - 1]

    # all the rows before
    if r1 > 0:
        result -= prefix[r1 - 1][c2]

    # we subtracted this square twice so we have to add it back
    if r1 > 0 and c1 > 0:
        result += prefix[r1 - 1][c1 - 1]

    return result

```

### Index as a hash key
If you are given a sequence and the interviewer asks for O(1) space, it might be possible to use the array itself as a hash table. For example, if the array only has values from 1 to N, where N is the length of the array, negate the value at that index (minus one) to indicate presence of that number. Examples: First Missing Positive, Daily Temperatures

### Traversing the array more than once
This might be obvious, but traversing the array twice/thrice (as long as fewer than n times) is still O(n). Sometimes traversing the array more than once can help you solve the problem while keeping the time complexity to O(n).

## String

### Counting characters
Often you will need to count the frequency of characters in a string. The most common way of doing that is by using a hash table/map in your language of choice. If your language has a built-in Counter class like Python, ask if you can use that instead.

If you need to keep a counter of characters, a common mistake is to say that the space complexity required for the counter is O(n). The space required for a counter of a string of latin characters is O(1) not O(n). This is because the upper bound is the range of characters, which is usually a fixed constant of 26. The input set is just lowercase Latin characters.

### Anagram
An anagram is word switch or word play. It is the result of rearranging the letters of a word or phrase to produce a new word or phrase, while using all the original letters only once. In interviews, usually we are only bothered with words without spaces in them.

To determine if two strings are anagrams, there are a few approaches:

- Sorting both strings should produce the same resulting string. This takes O(n.log(n)) time and O(log(n)) space.
- If we map each character to a prime number and we multiply each mapped number together, anagrams should have the same multiple (prime factor decomposition). This takes O(n) time and O(1) space. Examples: Group Anagram
- Frequency counting of characters will help to determine if two strings are anagrams. This also takes O(n) time and O(1) space.

### Palindrome
A palindrome is a word, phrase, number, or other sequence of characters which reads the same backward as forward, such as madam or racecar.

Here are ways to determine if a string is a palindrome:

- Reverse the string and it should be equal to itself.
- Have two pointers at the start and end of the string. Move the pointers inward till they meet. At every point in time, the characters at both pointers should match.
- The order of characters within the string matters, so hash tables are usually not helpful.

When a question is about counting the number of palindromes, a common trick is to have two pointers that move outward, away from the middle. Note that palindromes can be even or odd length. For each middle pivot position, you need to check it twice - once that includes the character and once without the character. This technique is used in Longest Palindromic Substring.

- For substrings, you can terminate early once there is no match
- For subsequences, use dynamic programming as there are overlapping subproblems


## Hash Table

- Describe an implementation of a least-used cache, and big-O notation of it.

Template
```
Initialize an LRU Cache with a given capacity:

1. Set cache capacity.
2. Create an empty hashmap that will hold key-value pairs (key -> node in doubly-linked list).
3. Create an empty doubly-linked list (nodes will have key-value pairs).

To GET a value from the cache:

1. If the key is not in the hashmap:
   a. Return "Not Found" or equivalent.
2. If the key is in the hashmap:
   a. Use the hashmap to get the node associated with that key from the doubly-linked list.
   b. Move the accessed node to the head of the doubly-linked list (indicating recent use).
   c. Return the value of the node.

To PUT a value in the cache:

1. If the key is already in the hashmap:
   a. Use the hashmap to get the node associated with that key from the doubly-linked list.
   b. Update the value of the node.
   c. Move the node to the head of the doubly-linked list.
2. If the key is not in the hashmap:
   a. Create a new node with the key-value pair.
   b. Add the node to the head of the doubly-linked list.
   c. Add the key and the node reference to the hashmap.
   d. If the size of the hashmap exceeds the cache capacity:
      i. Remove the node from the tail of the doubly-linked list (least recently used item).
      ii. Remove the corresponding key from the hashmap.

Helper Method - MoveToHead(node):

1. Remove the node from its current position in the doubly-linked list.
2. Add the node to the head of the doubly-linked list.
```

Example
```python
class ListNode:
    def __init__(self, key, val, nxt=None, prev=None):
        self.key = key
        self.val = val
        self.nxt = nxt
        self.prev = prev

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity

        self.left = ListNode(0,0)
        self.right = ListNode(0,0)
        self.right.prev = self.left
        self.left.nxt = self.right

    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.nxt = node
        nxt.prev = node
        node.prev = prev
        node.nxt = nxt

    def remove(self, node):
        prev, nxt = node.prev, node.nxt
        prev.nxt = nxt
        nxt.prev = prev

    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = ListNode(key, value)
        self.insert(self.cache[key])

        if len(self.cache) > self.capacity:
            evict = self.left.nxt
            self.remove(evict)
            del self.cache[evict.key]


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

## Permutations vs. Combinations vs. Power Set vs. Subset vs. Superset

These concepts help in counting and arranging elements from a set. Below is a simple breakdown.

---

### 1️⃣ Permutations (ORDER MATTERS)
**"How many ways can we ARRANGE things?"**
- Order **matters**
- No repeats (unless stated)

#### Example:
Given `{A, B, C}` and picking 2 letters:
- **AB, BA, AC, CA, BC, CB** (6 ways)

### Formula:
P(n, r) = n! / (n - r)!
- `n` = total items
- `r` = items chosen

**Use case:** Arranging people in a line, ranking in a competition.

---

### 2️⃣ Combinations (ORDER DOES NOT MATTER)
**"How many ways can we CHOOSE things?"**
- Order **does NOT** matter
- No repeats (unless stated)

#### Example:
Given `{A, B, C}` and picking 2 letters:
- **AB, AC, BC** (Just 3 ways, no BA, CA, or CB!)

#### Formula:
C(n, r) = n! / (r!(n - r)!)

**Use case:** Choosing lottery numbers, selecting a team.

---

### 3️⃣ Power Set (All Possible Subsets)
**"Give me every possible group of elements, including the empty set and full set."**
- Includes **all subsets** (even the empty set!)

#### Example:
Given `{A, B, C}`:
{ {}, {A}, {B}, {C}, {A, B}, {A, C}, {B, C}, {A, B, C} }


#### Formula:
For a set of size `n`, the power set has **`2^n` subsets**.
(E.g., `{A, B, C}` has `2^3 = 8` subsets.)

**Use case:** Finding every possible group of elements.

---

### 4️⃣ Subset (A "Part" of a Set)
**"A set that is fully contained in another set."**
- If `X` is a subset of `Y`, everything in `X` is in `Y`.
- **Every power set element is a subset**

#### Example:
- `{A, B}` is a subset of `{A, B, C}`, but `{A, D}` is NOT.

**Use case:** A folder inside another folder.

---

### 5️⃣ Superset (The "Bigger Set")
**"The opposite of a subset"**
- If `X` is a superset of `Y`, it contains all of `Y`.

#### Example:
- `{A, B, C}` is a superset of `{A, B}`.

**Use case:** The "parent" set.

---

### 6️⃣ Cartesian Product (Pairs from Two Sets)
**"All possible pairs between two sets"**
- If `A = {1, 2}` and `B = {X, Y}`, then:
A × B = {(1, X), (1, Y), (2, X), (2, Y)}


**Use case:** Matching items from two lists.

---

### **TL;DR Summary:**

| Concept | Order Matters? | Duplicates? | Example (for ABC) |
|---------|---------------|------------|-------------------|
| **Permutation** | ✅ Yes | ❌ No | `AB, BA, AC, CA, BC, CB` |
| **Combination** | ❌ No | ❌ No | `AB, AC, BC` |
| **Power Set** | ❌ No | ✅ Yes | `{ {}, {A}, {B}, {C}, {A, B}, {A, C}, {B, C}, {A, B, C} }` |
| **Subset** | ❌ No | ✅ Yes | `{A, B}` is a subset of `{A, B, C}` |
| **Superset** | ❌ No | ✅ Yes | `{A, B, C}` is a superset of `{A, B}` |
| **Cartesian Product** | ✅ Yes | ✅ Yes | `{(A, 1), (A, 2), (B, 1), (B, 2), (C, 1), (C, 2)}` |

---

### **Final Trick to Remember:**
✅ **If order matters → it's a permutation.**  
✅ **If order doesn't matter → it's a combination.**  
✅ **If you list ALL subsets → it's a power set.**  
✅ **If a set is inside another → it's a subset.**  
✅ **If a set contains another → it's a superset.**  


## Recursion

- Always remember to always define a base case so that your recursion will end.
- Recursion is useful for permutation, because it generates all combinations and tree-based questions. You should know how to generate all permutations of a sequence as well as how to handle duplicates.
- Recursion implicitly uses a stack. Hence all recursive approaches can be rewritten iteratively using a stack. Beware of cases where the recursion level goes too deep and causes a stack overflow (the default limit in Python is 1000). You may get bonus points for pointing this out to the interviewer. Recursion will never be O(1) space complexity because a stack is involved, unless there is tail-call optimization (TCO). Find out if your chosen language supports TCO.
- Number of base cases - In the fibonacci recursion, note that one of the recursive calls invoke fib(n - 2). This indicates that you should have 2 base cases defined so that your code covers all possible invocations of the function within the input range. If your recursive function only invokes fn(n - 1), then only one base case is needed.

### Memoization
In some cases, you may be computing the result for previously computed inputs. Let's look at the Fibonacci example again. fib(5) calls fib(4) and fib(3), and fib(4) calls fib(3) and fib(2). fib(3) is being called twice! If the value for fib(3) is memoized and used again, that greatly improves the efficiency of the algorithm and the time complexity becomes O(n).


## Binary search
When a given sequence is in a sorted order (be it ascending or descending), using binary search should be one of the first things that come to your mind.

Template
```
def binarySearch(array, target):
    1. Define two pointers: "left" initialized to 0 and "right" initialized to (length of array - 1).
    
    2. While "left" is less than or equal to "right":
       a. Calculate the middle index: mid = (left + right) / 2.
       b. If array[mid] equals target:
          i. Return mid.
       c. If array[mid] is less than target:
          i. Set left = mid + 1.
       d. Else:
          i. Set right = mid - 1.

    3. Return "Not Found" or an equivalent indicator (like -1).
```

```python
def binarySearch(arr: List, target: int):
    left = 0
    right = length(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

If you want to find the closest value that's less than the target value in a sorted array, you can modify the binary search algorithm slightly. At the end of the standard binary search loop, the right pointer will indicate the position where the target should be if it were in the array (or before it).

You can use this property to get the closest value less than the target. After the loop ends, the value at `arr[right]` would be the closest value less than the target (if right is within the bounds of the array).

```python
# While loop here
while ...

# Check if 'right' is within bounds
if right >= 0:
    return arr[right];

```

Other details:
- multiple binary searches are O(log(n) + log(m))
- sometimes you might have to compare mid with neighbouring values to find peaks or with left or right pointer to find a pivot 


## Matrix

### Create an empty X x M matrix:
When we say an 𝑋 × 𝑀, 𝑋 is typically the number of rows and 𝑀 is the number of columns. So 𝑋 × 𝑀 means 𝑋 rows and 𝑀 columns. For example, a 3 × 5 matrix has 3 rows and 5 columns.
```python
matrix = [[None] * M for _ in range(X)]
```

### Transposing a matrix
The transpose of a matrix is found by interchanging its rows into columns or columns into rows.

Many grid-based games can be modeled as a matrix, such as Tic-Tac-Toe, Sudoku, Crossword, Connect 4, Battleship, etc. It is not uncommon to be asked to verify the winning condition of the game. For games like Tic-Tac-Toe, Connect 4 and Crosswords, where verification has to be done vertically and horizontally, one trick is to write code to verify the matrix for the horizontal cells, transpose the matrix, and reuse the logic for horizontal verification to verify originally vertical cells (which are now horizontal).

```javascript
function transpose(matrix: number[][]): number[][] {
    // Create a new 2D array of size columns x rows of the original matrix
    const transposed: number[][] = Array.from({ length: matrix[0].length }, () => []);
    
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    
    return transposed;
}
```

### Linked List

### Sentinel/dummy nodes
Adding a sentinel/dummy node at the head and/or tail might help to handle many edge cases where operations have to be performed at the head or the tail. The presence of dummy nodes essentially ensures that operations will never be done on the head or the tail, thereby removing a lot of headache in writing conditional checks to dealing with null pointers. Be sure to remember to remove them at the end of the operation.

## Two pointers
Two pointer approaches are also common for linked lists. This approach is used for many classic linked list problems.

- Getting the kth from last node - Have two pointers, where one is k nodes ahead of the other. When the node ahead reaches the end, the other node is k nodes behind
- Detecting cycles - Have two pointers, where one pointer increments twice as much as the other, if the two pointers meet, means that there is a cycle
- Getting the middle node - Have two pointers, where one pointer increments twice as much as the other. When the faster node reaches the end of the list, the slower node will be at the middle. We can have `while curr and curr.next` to stop after middle in even length linked list or we can add `and curr.next.next` to stop before middle.

Example - where multiple pointers

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

Input: head = [1,2,3,4]

Output: [2,1,4,3]

Input: head = [1,2,3]

Output: [2,1,3]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        dummy = ListNode(0, head)
        # Why prev? When we iterate through the list we need to set the prev pointer to the second node in the current pair
        prev, curr = dummy, head

        while curr and curr.next:
            # identify nodes of importance. It is often useful to simplify pointers doing what we do with `second`. It makes future executions with second much easier.
            next_pair = curr.next.next
            second = curr.next

            # do the swaps and make prev point to second
            second.next = curr
            curr.next = next_pair
            prev.next = second

            # update the pointers
            prev = curr
            curr = next_pair

        return dummy.next
```

In the `swapPairs` function, we utilize a dummy node as the head of our list. By doing this, we don't have to write special logic to initialize the head of the list, because `dummy.next` will naturally point to the start of the merged list at the end of the process. The dummy node serves as a placeholder and helps in simplifying the code. It also provides an initial `prev.next` for `second`.

### Using space
Many linked list problems can be easily solved by creating a new linked list and adding nodes to the new linked list with the final result. However, this takes up extra space and makes the question much less challenging. The interviewer will usually request that you modify the linked list in-place and solve the problem without additional storage. You can borrow ideas from the Reverse a Linked List problem.

### Elegant modification operations
As mentioned earlier, a linked list's non-sequential nature of memory allows for efficient modification of its contents. Unlike arrays where you can only modify the value at a position, for linked lists you can also modify the next pointer in addition to the value.

Here are some common operations and how they can be achieved easily:
- Truncate a list - Set the next pointer to null at the last element
- Swapping values of nodes - Just like arrays, just swap the value of the two nodes, there's no need to swap the next pointer
- Combining two lists - attach the head of the second list to the tail of the first list


## Interval

### Sort the array of intervals by its starting point
A common routine for interval questions is to sort the array of intervals by each interval's starting value. This step is crucial to solving the Merge Intervals question.

### Checking if two intervals overlap
Be familiar with writing code to check if two intervals overlap.

```python
def is_overlap(a, b):
  return a[0] < b[1] and b[0] < a[1]
```
Trick to remember: both the higher pos must be greater then both lower pos.

Merging two intervals
```python
def merge_overlapping_intervals(a, b):
  return [min(a[0], b[0]), max(a[1], b[1])]
```

## Tree

![Tree](https://upload.wikimedia.org/wikipedia/commons/5/5e/Binary_tree_v2.svg)

### Traversals

- Inorder traversal - Left -> Root -> Right.<br/>
Result: 2, 7, 5, 6, 11, 1, 9, 5, 9
- Preorder traversal - Root -> Left -> Right<br/>
Result: 1, 7, 2, 6, 5, 11, 9, 9, 5
- Postorder traversal - Left -> Right -> Root<br/>
Result: 2, 5, 11, 6, 7, 5, 9, 9, 1

Example (Recursive)
```python
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
```
```python
# Inorder Traversal (Left, Root, Right)
def inorder_traversal(root):
    res = []

    def helper(root):
        if not root:
            return None

        if root.left:
            inorder(root.left)

        res.append(root.val)

        if root.right:
            inorder(root.right)

    helper(root)
    return res
```
Iterative inorder traversal

```python
def iterative_inorder_traversal(root):
    res = []
    stack = []
    curr = root

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        
        res.append(curr.val)
        
        if curr.right:
            curr = curr.right

    return res
```

```python
# Pre-Order Traversal (Root, Left, Right)
def preOrderTraversal(root: TreeNode):
    result = []

    def helper(node: TreeNode):
        if node == None: 
            return

        result.append(node.val)
        helper(node.left)
        helper(node.right)

    helper(root)
    return result
```


```python
# Post-Order Traversal (Left, Right, Root)
def postOrderTraversal(root: TreeNode):
    result = []

    def helper(node: TreeNode):
        if node == None: 
            return

        helper(node.left)
        helper(node.right)
        result.append(node.val)

    helper(root)
    return result

```

Example (Iterative)
```javascript
class TreeNode {
    val: number;
    left: TreeNode | null = null;
    right: TreeNode | null = null;

    constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
        this.val = (val === undefined ? 0 : val);
        this.left = (left === undefined ? null : left);
        this.right = (right === undefined ? null : right);
    }
}


// Pre-Order Traversal (Root, Left, Right)
function preOrderTraversal(root: TreeNode | null): number[] {
    const result: number[] = [];
    const stack: TreeNode[] = [];
    if (root) stack.push(root);

    while (stack.length > 0) {
        const current = stack.pop()!;
        result.push(current.val);

        if (current.right) stack.push(current.right);
        if (current.left) stack.push(current.left);
    }

    return result;
}

// Post-Order Traversal (Left, Right, Root)
function postOrderTraversal(root: TreeNode | null): number[] {
    const result: number[] = [];
    const stack: TreeNode[] = [];
    let lastVisited: TreeNode | null = null;

    while (root || stack.length > 0) {
        while (root) {
            stack.push(root);
            root = root.left;
        }

        const peekNode = stack[stack.length - 1];

        if (!peekNode.right || peekNode.right === lastVisited) {
            result.push(peekNode.val);
            lastVisited = stack.pop()!;
        } else {
            root = peekNode.right;
        }
    }

    return result;
}
```

### Binary search tree (BST)
In-order traversal of a BST will give you all elements in order.

Be very familiar with the properties of a BST and validating that a binary tree is a BST. This comes up more often than expected.

When a question involves a BST, the interviewer is usually looking for a solution which runs faster than O(n).

**Time complexity**<br/>
Operation	Big-O<br/>
Access	O(log(n))<br/>
Search	O(log(n))<br/>
Insert	O(log(n))<br/>
Remove	O(log(n))

Space complexity of traversing balanced trees is O(h) where h is the height of the tree, while traversing very skewed trees (which is essentially a linked list) will be O(n).

### Use recursion
Recursion is the most common approach for traversing trees. When you notice that the subtree problem can be used to solve the entire problem, try using recursion.

When using recursion, always remember to check for the base case, usually where the node is `null`.

Sometimes it is possible that your recursive function needs to return two values.

### Traversing by level
When you are asked to traverse a tree by level, use breadth-first search.

## Graph

### Graph representations
You can be given a list of edges and you have to build your own graph from the edges so that you can perform a traversal on them. The common graph representations are:

- Adjacency matrix
- Adjacency list
- Hash table of hash tables

Using a hash table of hash tables would be the simplest approach during algorithm interviews. It will be rare that you have to use an adjacency matrix or list for graph questions during interviews.

Graph problems usually fall into one of the following categories:

### 1. Has a Cycle? (Cycle Detection)
#### **Type:** Cycle Detection (Directed or Undirected)
#### **Approaches:**
- **Undirected Graph:** Use **DFS with a parent check** or **Union-Find**.
- **Directed Graph:** Use **DFS with a visited + recursion stack** check (detects back edges).
- **Kahn’s Algorithm (BFS-based Topological Sort):** If all nodes are processed but a node still has an incoming edge, a cycle exists.

#### **Example Problems:**
- Detect if an undirected graph has a cycle.
- Detect if a directed graph has a cycle (i.e., check if it’s a Directed Acyclic Graph, DAG).

---

### 2. Has a Valid Path? (Path Finding / Connectivity)
#### **Type:** Path Finding / Connectivity Check
#### **Approaches:**
- **BFS / DFS:** Standard traversal to check if two nodes are connected.
- **Union-Find (Disjoint Set):** If both nodes belong to the same connected component, a path exists.
- **Dijkstra’s Algorithm:** If weighted, find the shortest path.
- **Bellman-Ford / Floyd-Warshall:** If there are negative weights.

#### **Example Problems:**
- Find if there is a path between two nodes.
- Find the shortest path between two nodes.

---

### 3. What is the Path? (Topological Sorting / Order of Execution)
#### **Type:** Topological Sorting (for Directed Acyclic Graphs - DAGs)
#### **Approaches:**
- **DFS-Based Topological Sort:** Perform DFS and push nodes to a stack (reverse order gives topological sorting).
- **Kahn’s Algorithm (BFS-Based Topological Sort):** Maintain **in-degree** counts and process nodes with in-degree = 0.

#### **Example Problems:**
- Course schedule problems (can you take all courses in order?).
- Build systems / Task scheduling with dependencies.
- Finding a valid order of execution.

---

### 4. What is the Shortest / Cheapest Path? (Weighted Graphs)
#### **Type:** Shortest Path Algorithms
#### **Approaches:**
- **Unweighted Graph:** BFS finds the shortest path.
- **Weighted Graph (No Negative Weights):** Dijkstra’s Algorithm.
- **Weighted Graph (Negative Weights Allowed):** Bellman-Ford Algorithm.
- **Find shortest paths between all pairs:** Floyd-Warshall Algorithm.

#### **Example Problems:**
- Find the shortest route between two cities.
- Optimize network packet transmission.

---

### 5. What are the Connected Components? (Find All Groups of Connected Nodes)
#### **Type:** Connected Components / Graph Clustering
#### **Approaches:**
- **DFS / BFS:** Find and mark all reachable nodes in one traversal.
- **Union-Find:** Keep track of connected components dynamically.

#### **Example Problems:**
- Count the number of provinces (groups of friends).
- Find islands in a 2D grid (number of connected land regions).

---

### 6. Is the Graph Bipartite? (Two-Coloring Problem)
#### **Type:** Graph Coloring / Bipartite Check
#### **Approaches:**
- **BFS / DFS with two colors:** Alternate colors while traversing.
- **Union-Find:** If two nodes in the same component have the same color, it’s not bipartite.

#### **Example Problems:**
- Check if a given graph is bipartite.
- Determine if a graph can be divided into two groups without conflicts.

---

### Final Thoughts
- If a problem involves finding **connections**, **paths**, or **cycles**, it’s a graph problem.
- **BFS** is great for **shortest unweighted paths**.
- **DFS** is useful for **cycle detection**, **topological sorting**, and **exploring all paths**.
- **Union-Find** is best for **connectivity problems**.
- **Dijkstra and Bellman-Ford** help with **weighted shortest path problems**.


```javascript
// Let's assume you're given a list of edges as input, where each edge is a tuple of two nodes, e.g., ["A", "B"] means there's an edge between nodes "A" and "B"

type Node = string;
type Graph = Map<Node, Map<Node, boolean>>;

function addEdge(graph: Graph, from: Node, to: Node) {
    if (!graph.has(from)) {
        graph.set(from, new Map<Node, boolean>());
    }
    graph.get(from)!.set(to, true);
}

function buildGraph(edges: [Node, Node][]): Graph {
    const graph = new Map<Node, Map<Node, boolean>>();

    for (const [from, to] of edges) {
        addEdge(graph, from, to);
        addEdge(graph, to, from); // For undirected graph. Remove this for directed graph.
    }

    return graph;
}

// Test
const edges: [Node, Node][] = [
    ["A", "B"],
    ["A", "C"],
    ["B", "D"],
    ["C", "D"],
    ["D", "E"]
];

const graph = buildGraph(edges);

console.log(graph);
```

In algorithm interviews, graphs are commonly given in the input as 2D matrices where cells are the nodes and each cell can traverse to its adjacent cells (up/down/left/right). Hence it is important that you be familiar with traversing a 2D matrix. When traversing the matrix, always ensure that your current position is within the boundary of the matrix and has not been visited before.

```javascript
type Matrix = number[][];


// O(rows * cols) -> time and space
function traverseMatrix(matrix: Matrix): void {
    if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
        return;
    }

    const rows = matrix.length;
    const cols = matrix[0].length;
    const visited: boolean[][] = Array.from({ length: rows }, () => Array(cols).fill(false));

    function isValid(row: number, col: number): boolean {
        return row >= 0 && row < rows && col >= 0 && col < cols && !visited[row][col];
    }

    function dfs(row: number, col: number): void {
        if (!isValid(row, col)) {
            return;
        }

        console.log(matrix[row][col]); // Process the current cell
        visited[row][col] = true;

        // Traverse up/down/left/right
        dfs(row - 1, col); // Up
        dfs(row + 1, col); // Down
        dfs(row, col - 1); // Left
        dfs(row, col + 1); // Right
    }

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (!visited[i][j]) {
                dfs(i, j);
            }
        }
    }
}

// Test
const matrix: Matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];

traverseMatrix(matrix);
```

### 2D Matrix as a graph

Transforming the problem of traversing a 2D matrix into a graph problem involves viewing each cell of the matrix as a node of a graph and the possible movements (e.g., up, down, left, right) from a cell as edges connecting the nodes.

Here's a step-by-step guide on how you can transform the problem:

1. **Nodes Representation**: Each cell in the matrix, identified by its coordinates (i, j), can be treated as a node in the graph.

2. **Edges Representation**: For each cell, consider its adjacent cells. An edge exists between two nodes if you can move from one cell to the adjacent cell. For instance, if movements are restricted to up, down, left, and right, then:
   - The node (i, j) will have an edge to (i-1, j) if (i-1, j) is a valid cell (i.e., moving upwards).
   - The node (i, j) will have an edge to (i+1, j) if (i+1, j) is a valid cell (i.e., moving downwards).
   - The node (i, j) will have an edge to (i, j-1) if (i, j-1) is a valid cell (i.e., moving left).
   - The node (i, j) will have an edge to (i, j+1) if (i, j+1) is a valid cell (i.e., moving right).

3. **Graph Representation**: There are several ways to represent a graph, such as an adjacency list, adjacency matrix, or a hash table of hash tables. In the context of a matrix traversal, an adjacency list is often a good fit. Each node (i.e., matrix cell) would have a list of its adjacent nodes.

4. **Traversal**: With the graph constructed, you can apply standard graph traversal algorithms like DFS or BFS to explore the nodes. 

5. **Special Conditions**: If there are certain cells in the matrix that you cannot traverse (like obstacles), you simply skip adding them as valid edges in the graph representation. 

This transformation is particularly useful when you have more complex conditions for traversal, or when the problem involves finding paths, connected components, etc. It allows you to leverage a wide range of graph algorithms to solve matrix-based problems.

Example
```javascript
type Node = [number, number];
type Graph = Map<string, Node[]>;

function matrixToGraph(matrix: number[][]): Graph {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const graph: Graph = new Map();

    function getNodeKey(node: Node): string {
        return `${node[0]},${node[1]}`;
    }

    function getAdjacentNodes(row: number, col: number): Node[] {
        const directions: Node[] = [[-1, 0], [1, 0], [0, -1], [0, 1]];  // Up, Down, Left, Right
        const adjacentNodes: Node[] = [];

        for (const [dx, dy] of directions) {
            const newRow = row + dx;
            const newCol = col + dy;

            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
                adjacentNodes.push([newRow, newCol]);
            }
        }

        return adjacentNodes;
    }

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const node: Node = [i, j];
            const key = getNodeKey(node);
            graph.set(key, getAdjacentNodes(i, j));
        }
    }

    return graph;
}

function traverseGraph(graph: Graph, startNode: Node): void {
    const visited: Set<string> = new Set();

    function dfs(node: Node): void {
        const key = `${node[0]},${node[1]}`;
        if (visited.has(key)) return;

        console.log(node);
        visited.add(key);

        const neighbors = graph.get(key) || [];
        for (const neighbor of neighbors) {
            dfs(neighbor);
        }
    }

    dfs(startNode);
}

// Test
const matrix: number[][] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];

const graph = matrixToGraph(matrix);
traverseGraph(graph, [0, 0]);  // Start traversal from top-left corner
```

### Time complexity
|V| is the number of vertices while |E| is the number of edges.

Algorithm	Big-O<br/>
Depth-first search	O(|V| + |E|)<br/>
Breadth-first search	O(|V| + |E|)<br/>
Topological sort	O(|V| + |E|)<br/>

### Notes
A tree-like diagram could very well be a graph that allows for cycles and a naive recursive solution would not work. In that case you will have to handle cycles and keep a set of visited nodes when traversing.

Ensure you are correctly keeping track of visited nodes and not visiting each node more than once. Otherwise your code could end up in an infinite loop.

### Depth-first search

```python
def dfs(matrix):
  # Check for an empty matrix/graph.
  if not matrix:
    return []

  rows, cols = len(matrix), len(matrix[0])
  visited = set()
  directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

  def traverse(i, j):
    if (i, j) in visited:
      return

    visited.add((i, j))

    # Traverse neighbors.
    for direction in directions:
      next_i, next_j = i + direction[0], j + direction[1]
      if 0 <= next_i < rows and 0 <= next_j < cols:

        # Add in question-specific checks, where relevant.
        traverse(next_i, next_j)

  for i in range(rows):
    for j in range(cols):
      traverse(i, j)
```

### Breadth-first search

```python
from collections import deque

def bfs(matrix):
  # Check for an empty matrix/graph.
  if not matrix:
    return []

  rows, cols = len(matrix), len(matrix[0])
  visited = set()
  directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

  def traverse(i, j):
    queue = deque([(i, j)])
    while queue:
      curr_i, curr_j = queue.popleft()

      if (curr_i, curr_j) not in visited:
        visited.add((curr_i, curr_j))

        # Traverse neighbors.
        for direction in directions:
          next_i, next_j = curr_i + direction[0], curr_j + direction[1]
          if 0 <= next_i < rows and 0 <= next_j < cols:

            # Add in question-specific checks, where relevant.
            queue.append((next_i, next_j))

  for i in range(rows):
    for j in range(cols):
      traverse(i, j)
```

### Topological sorting

A topological sort or topological ordering of a directed graph is a linear ordering of its vertices such that for every directed edge uv from vertex u to vertex v, u comes before v in the ordering. Precisely, a topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.

Topological sorting is most commonly used for scheduling a sequence of jobs or tasks which has dependencies on other jobs/tasks. The jobs are represented by vertices, and there is an edge from x to y if job x must be completed before job y can be started.

Another example is taking courses in university where courses have pre-requisites.

In the context of directed graphs, in-degree of a node refers to the number of incoming edges to that node. In simpler terms, it's a count of how many vertices "point to" or "lead to" the given vertex.

```javascript
type Graph = Map<number, number[]>;

function topologicalSort(graph: Graph): number[] | null {
    const numNodes = graph.size;
    const inDegree: Map<number, number> = new Map();
    const zeroInDegreeQueue: number[] = [];
    const result: number[] = [];

    // Initialize in-degree counts
    for (const [node, neighbors] of graph.entries()) {
        inDegree.set(node, 0);
    }

    for (const [, neighbors] of graph.entries()) {
        for (const neighbor of neighbors) {
            inDegree.set(neighbor, (inDegree.get(neighbor) || 0) + 1);
        }
    }

    // Find all nodes with zero in-degree
    for (const [node, degree] of inDegree.entries()) {
        if (degree === 0) {
            zeroInDegreeQueue.push(node);
        }
    }

    while (zeroInDegreeQueue.length) {
        const current = zeroInDegreeQueue.shift()!;
        result.push(current);

        const neighbors = graph.get(current) || [];
        for (const neighbor of neighbors) {
            inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
            if (inDegree.get(neighbor) === 0) {
                zeroInDegreeQueue.push(neighbor);
            }
        }
    }

    if (result.length !== numNodes) {
        return null;  // Graph has a cycle
    }

    return result;
}

// Test
const graph: Graph = new Map();
graph.set(5, [2]);
graph.set(4, [0, 2]);
graph.set(2, [3]);
graph.set(3, [1]);
graph.set(1, []);
graph.set(0, []);

const sortedOrder = topologicalSort(graph);
console.log(sortedOrder);  // Possible output: [5, 4, 2, 3, 1, 0]
```

## Heap

A heap is a specialized tree-based data structure which is a complete tree that satisfies the heap property.

- Max heap - In a max heap, the value of a node must be greatest among the node values in its entire subtree. The same property must be recursively true for all nodes in the tree.
- Min heap - In a min heap, the value of a node must be smallest among the node values in its entire subtree. The same property must be recursively true for all nodes in the tree.

In the context of algorithm interviews, heaps and priority queues can be treated as the same data structure. A heap is a useful data structure when it is necessary to repeatedly remove the object with the highest (or lowest) priority, or when insertions need to be interspersed with removals of the root node.

### Time complexity
Operation	Big-O<br/>
Find max/min	O(1)<br/>
Insert	O(log(n))<br/>
Remove	O(log(n))<br/>
Heapify (create a heap out of given array of elements)	O(n)<br/>

### Mention of k
If you see a top or lowest k being mentioned in the question, it is usually a signal that a heap can be used to solve the problem, such as in Top K Frequent Elements.

If you require the top k elements use a Min Heap of size k. Iterate through each element, pushing it into the heap (for python heapq, invert the value before pushing to find the max). Whenever the heap size exceeds k, remove the minimum element, that will guarantee that you have the k largest elements.

## Trie

Tries are special trees (prefix trees) that make searching and storing strings more efficient. Tries have many practical applications, such as conducting searches and providing autocomplete. It is helpful to know these common applications so that you can easily identify when a problem can be efficiently solved using a trie.

Be familiar with implementing from scratch, a Trie class and its add, remove and search methods.

Example:
```python
class TrieNode:
    def __init__(key, eow):
        self.key = key
        self.isendofword = 
class TrieNode {
    children: { [key: string]: TrieNode } = {};
    isEndOfWord: boolean = false;
}

class Trie {
    private root: TrieNode = new TrieNode();

    // Inserts a word into the trie
    insert(word: string): void {
        let currentNode = this.root;
        for (let char of word) {
            if (!currentNode.children[char]) {
                currentNode.children[char] = new TrieNode();
            }
            currentNode = currentNode.children[char];
        }
        currentNode.isEndOfWord = true;
    }

    // Returns if the word is in the trie
    search(word: string): boolean {
        let currentNode = this.root;
        for (let char of word) {
            if (!currentNode.children[char]) {
                return false;
            }
            currentNode = currentNode.children[char];
        }
        return currentNode.isEndOfWord;
    }

    // Returns if there's any word in the trie that starts with the given prefix
    startsWith(prefix: string): boolean {
        let currentNode = this.root;
        for (let char of prefix) {
            if (!currentNode.children[char]) {
                return false;
            }
            currentNode = currentNode.children[char];
        }
        return true;
    }
}

// Test
const trie = new Trie();
trie.insert("apple");
console.log(trie.search("apple"));    // Expected output: true
console.log(trie.search("app"));      // Expected output: false
console.log(trie.startsWith("app"));  // Expected output: true
trie.insert("app");
console.log(trie.search("app"));      // Expected output: true
```

### Time complexity
`m` is the length of the string used in the operation.

Operation	Big-O<br/>
Search	O(m)<br/>
Insert	O(m)<br/>
Remove	O(m)<br/>

### Techniques
Sometimes preprocessing a dictionary of words (given in a list) into a trie, will improve the efficiency of searching for a word of length k, among n words. Searching becomes O(k) instead of O(n).

## Geometry

Distance between two points
```javascript
type Point = {
    x: number,
    y: number
};

function distanceBetweenTwoPoints(point1: Point, point2: Point): number {
    return Math.sqrt(Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2));
}
```

Overlapping Circles
```javascript
type Circle = {
    center: Point,
    radius: number
};

function areCirclesOverlapping(circle1: Circle, circle2: Circle): boolean {
    const distance = distanceBetweenTwoPoints(circle1.center, circle2.center);
    return distance <= (circle1.radius + circle2.radius);
}
```

Overlapping Rectanges
```javascript
type Rectangle = {
    topLeft: Point,
    bottomRight: Point
};

function areRectanglesOverlapping(rect1: Rectangle, rect2: Rectangle): boolean {
    // Check if one rectangle is to the left of the other
    if (rect1.topLeft.x > rect2.bottomRight.x || rect2.topLeft.x > rect1.bottomRight.x) {
        return false;
    }

    // Check if one rectangle is above the other
    if (rect1.topLeft.y < rect2.bottomRight.y || rect2.topLeft.y < rect1.bottomRight.y) {
        return false;
    }

    return true; // If none of the above cases occurred, rectangles are overlapping
}
```

# Notorious Problems

## Robot Room Cleaner
Link: https://leetcode.com/problems/robot-room-cleaner/description/

```javascript
/**
 * class Robot {
 *      // Returns true if the cell in front is open and robot moves into the cell.
 *      // Returns false if the cell in front is blocked and robot stays in the current cell.
 * 		move(): boolean {}
 * 		
 *      // Robot will stay in the same cell after calling turnLeft/turnRight.
 *      // Each turn will be 90 degrees.
 * 		turnRight() {}
 * 		
 *      // Robot will stay in the same cell after calling turnLeft/turnRight.
 *      // Each turn will be 90 degrees.
 * 		turnLeft() {}
 * 		
 * 		// Clean the current cell.
 * 		clean(): {}
 * }
 */

function cleanRoom(robot: Robot) {
    const dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    const visited = new Set();

    const goBack = () => {
        robot.turnRight();
        robot.turnRight();
        robot.move();
        robot.turnRight();
        robot.turnRight();
    }

    const backtrack = (row: number, col: number, d: number) => {
        visited.add([row, col].join());
        robot.clean();

        for (let i = 0; i < 4; ++i) {
            const newD = (d + i) % 4;
            const [newRow, newCol] = [row + dirs[newD][0], col + dirs[newD][1]];

            if (!visited.has([newRow, newCol].join()) && robot.move()) {
                backtrack(newRow, newCol, newD);
                goBack();
            }

            robot.turnRight();
        }
    }

    backtrack(0, 0, 0);
};
```

## Kadane's Algorithm

Kadane's algorithm is used to find the maximum sum of a contiguous subarray within a one-dimensional array of numbers.

It checks the first contiguous numbers and if they are below 0, we can ignore those coniguous numbers

```python
def maxSubArraySum(nums):
    if len(nums) == 0:
        return 0

    tot = 0
    max_tot = -float("inf")

    for i in range(len(nums)):
        if tot < 0:
            tot = 0

        tot += nums[i]
        max_tot = max(max_tot, tot)

    return max_tot


# Example usage:
numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(maxSubArraySum(numbers))  # Outputs: 6 (because [4, -1, 2, 1] has the largest sum)
```

If you a circular subarray (the end is connected to the start), the intuition is to find the min contigious sub array. The remaining elements would be the max contigious sub array so you get sum(nums) - min_tot. This has only a chance of being a solution. If the actual max subarray isn't circular, then this won't work so in this case, use the usual algorithm.

There is an edge case where all the numbers are negative:
```python
nums = [-3,-2,-3]
```
The smallest possible contiguous array is all of the elements which is the same as the sum of the array.

Thus sum_array - min_tot = 0 which is larger than max_tot which is not the right answer.

The right answer is just max_tot
```python
def maxSubarraySumCircular(self, nums: List[int]) -> int:
    if not nums:
        return 0
    
    max_tot = -float("inf")
    min_tot = float("inf")
    sum_array = 0
    curr_max = 0
    curr_min = 0
    
    for num in nums:
        # to find total sum
        sum_array += num
        
        # For maximum subarray
        if curr_max < 0:
            curr_max = 0
        curr_max += num
        max_tot = max(max_tot, curr_max)
        
        # For minimum subarray
        if curr_min > 0:
            curr_min = 0
        curr_min += num
        min_tot = min(min_tot, curr_min)
    
    return max(max_tot, sum_array - min_tot) if max_tot > 0 else max_tot
```

## Great Explanation of Dynamic Programming (DP)

- https://leetcode.com/problems/word-break-ii/editorial/

## Disjoint Union Set (DSU)

```javascript
/*
Return the lexicographically smallest string that s can be changed to after using the swaps.

Input: s = "dcab", pairs = [[0,3],[1,2]]
Swap s[0] and s[3], s = "bcad"
Swap s[1] and s[2], s = "bacd"

Input: s = "cba", pairs = [[0,1],[1,2]]
Swap s[0] and s[1], s = "bca"
Swap s[1] and s[2], s = "bac"
Swap s[0] and s[1], s = "abc"

Input: s = "cbfdae", pairs = [[0,1],[3,2],[5,2],[1,4]]
Output: "abdecf"
*/
function smallestStringWithSwaps(s: string, pairs: number[][]): string {
    const parent: number[] = [];
    for (let i = 0; i < s.length; i++) {
        parent[i] = i;
    }

    // Find the parent of a node
    function find(i: number): number {
        if (i !== parent[i]) {
            parent[i] = find(parent[i]);
        }
        return parent[i];
    }

    // Union the two nodes
    function union(i: number, j: number) {
        parent[find(i)] = find(j);
    }

    // Create the disjoint set using the pairs
    for (let [i, j] of pairs) {
        union(i, j);
    }

    // Create mapping of root -> [indices]
    const indexGroups: { [key: number]: number[] } = {};
    for (let i = 0; i < s.length; i++) {
        const root = find(i);
        if (!indexGroups[root]) {
            indexGroups[root] = [];
        }
        indexGroups[root].push(i);
    }

    let chars: string[] = s.split('');
    for (let indices of Object.values(indexGroups)) {
        const sortedChars = indices.map(i => chars[i]).sort();
        for (let i = 0; i < indices.length; i++) {
            chars[indices[i]] = sortedChars[i];
        }
    }

    return chars.join('');
}

// Test cases
console.log(smallestStringWithSwaps("dcab", [[0, 3], [1, 2]])); // "bacd"
console.log(smallestStringWithSwaps("cba", [[0, 1], [1, 2]]));  // "abc"
console.log(smallestStringWithSwaps("cbfdae", [[0, 1], [3, 2], [5, 2], [1, 4]])); // "abdecf"

```

Let's dry run the `smallestStringWithSwaps` function with the input string `s = "dcab"` and pairs `[[0,3],[1,2]]`.

**Initialization**:
```typescript
const parent: number[] = [];
for (let i = 0; i < s.length; i++) {
    parent[i] = i;
}
```
This initializes each character's index as its own parent:
`parent = [0, 1, 2, 3]`

**Processing the pairs using union**:

1. For the pair `[0,3]`:
   - The union function merges the sets containing `0` and `3`.
   - `parent` becomes: `[3, 1, 2, 3]`

2. For the pair `[1,2]`:
   - The union function merges the sets containing `1` and `2`.
   - `parent` becomes: `[3, 2, 2, 3]`

**Create mapping of root to its indices**:
We want to know which indices belong to which set. 
```typescript
const indexGroups: { [key: number]: number[] } = {};
for (let i = 0; i < s.length; i++) {
    const root = find(i);
    if (!indexGroups[root]) {
        indexGroups[root] = [];
    }
    indexGroups[root].push(i);
}
```
After this loop, `indexGroups` becomes:
```
{
    3: [0, 3],
    2: [1, 2]
}
```
This means characters at indices `0` and `3` can be swapped with each other, and characters at indices `1` and `2` can be swapped with each other.

**Reorder characters within each set**:
We want the characters in each group to be sorted in ascending order to make the string lexicographically smallest.
```typescript
let chars: string[] = s.split('');
for (let indices of Object.values(indexGroups)) {
    const sortedChars = indices.map(i => chars[i]).sort();
    for (let i = 0; i < indices.length; i++) {
        chars[indices[i]] = sortedChars[i];
    }
}
```
1. For `indices = [0, 3]` (i.e., characters `'d'` and `'b'`):
   - Sort them to get `'b'` and `'d'`.
   - Update the characters in `chars` to reflect the new order: `chars` becomes `['b', 'c', 'a', 'd']`.

2. For `indices = [1, 2]` (i.e., characters `'c'` and `'a'`):
   - Sort them to get `'a'` and `'c'`.
   - Update the characters in `chars` to reflect the new order: `chars` becomes `['b', 'a', 'c', 'd']`.

**Return the modified string**:
The final step is to join the `chars` array:
```typescript
return chars.join('');
```
This returns the string `"bacd"`.

So, for the input `s = "dcab"` and pairs `[[0,3],[1,2]]`, the function `smallestStringWithSwaps` returns the string `"bacd"`.

## Monotonic Stack

A monotonic stack is a type of stack that maintains a certain monotonicity property as elements are pushed into it. Depending on the problem at hand, this might be a strictly increasing or strictly decreasing stack. In other words, when you push a new element onto the stack, the stack ensures that its elements are still in either non-decreasing or non-increasing order by potentially popping off the elements that violate the monotonicity.

The primary purpose of a monotonic stack is to efficiently answer questions like:

- For each element in an array, what's the nearest smaller (or larger) element on the left or right of it?
- Finding the maximum rectangle in a histogram.

Let's dive into an example:

**Problem Statement:**
Given a circular array of numbers, for each element, find the first larger number to its right. If no larger number return -1.

**Example:**
Input: `[4, 3, 2, 5, 1]`
Output: `[5, 5, 5, -1, 4]`

**Explanation:** 
For `4`, the next larger number is `5`.
For `3`, the next larger number is `5`.
For `2`, the next larger number is `5`.
For `5`, there is no larger number, hence `-1`.
For `1`, the next larger number is `4`.

**Python Solution using Monotonic Stack:**
```python
def nextLargerElement(nums: List[int]) -> List[int]

    result = [-1] * len(nums) # initialize result array with -1 as we want to return -1 if no next max
    stack = []

    for i in range(len(nums)*2): # multiply by 2 to simulate traversing circular array
        while stack and nums[i % len(nums)] > nums[stack[-1]]:
            top = stack.pop()
            res[top] = nums[i % len(nums)]


        if i < len(nums):
            stack.append(i)

    return result


# Testing the function
input = [4, 3, 2, 5, 1]
print(nextLargerElement(input)) # Expected output: [5, 5, 5, -1, 4]
```

In this solution, we're keeping the stack in non-decreasing order. When a larger number is found, we keep popping from the stack until we find a number that's greater than the current one or the stack becomes empty. This helps in finding the next larger number for all the numbers that are smaller than the current number.

## Monotonic queues

When to use this? When we wanna find out about a property in a partrticular range or sliding window. We are now using deques to identify a min or max (could also use heap but popleft() and pop() are better than heap operations)

**Problem Statement:**
Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that the absolute difference between any two elements of this subarray is less than or equal to limit.
**Example:**
nums = `[8,2,4,7]` , limit = 4
Output: 2

**Explanation:** 
All subarrays are: 
- [8] with maximum absolute diff |8-8| = 0 <= 4.
- [8,2] with maximum absolute diff |8-2| = 6 > 4. 
- [8,2,4] with maximum absolute diff |8-2| = 6 > 4.
- [8,2,4,7] with maximum absolute diff |8-2| = 6 > 4.
- [2] with maximum absolute diff |2-2| = 0 <= 4.
- [2,4] with maximum absolute diff |2-4| = 2 <= 4.
- [2,4,7] with maximum absolute diff |2-7| = 5 > 4.
- [4] with maximum absolute diff |4-4| = 0 <= 4.
- [4,7] with maximum absolute diff |4-7| = 3 <= 4.
- [7] with maximum absolute diff |7-7| = 0 <= 4.

 Therefore, the size of the longest subarray is 2.

 *Python Solution using Monotonic Stack:**
```python
def longestSubarray(self, nums: List[int], limit: int) -> int:

        l = 0
        max_queue = deque()  # Stores indices in decreasing order (max values)
        min_queue = deque()  # Stores indices in increasing order (min values)
        res = 0

        for r in range(len(nums)):
            # Maintain decreasing order in max_queue
            while max_queue and nums[max_queue[-1]] < nums[r]:
                max_queue.pop()
            max_queue.append(r)

            # Maintain increasing order in min_queue
            while min_queue and nums[min_queue[-1]] > nums[r]:
                min_queue.pop()
            min_queue.append(r)

            # Shrink window if the difference exceeds limit
            while nums[max_queue[0]] - nums[min_queue[0]] > limit:
                if l > max_queue[0]:  # If the max element index is out of bounds
                    max_queue.popleft()
                if l > min_queue[0]:  # If the min element index is out of bounds
                    min_queue.popleft()
                l += 1  # Shrink the window from the left

            res = max(res, r - l + 1)  # Update longest valid subarray length

        return res


```

## Peak Valley (Arrays)

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

```
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
```

```javascript
class Solution {
    public int maxProfit(int[] prices) {
        int i = 0;
        int valley = prices[0];
        int peak = prices[0];
        int maxprofit = 0;
        while (i < prices.length - 1) {
            while (i < prices.length - 1 && prices[i] >= prices[i + 1])
                i++;
            valley = prices[i];
            while (i < prices.length - 1 && prices[i] <= prices[i + 1])
                i++;
            peak = prices[i];
            maxprofit += peak - valley;
        }
        return maxprofit;
    }
}
```

## Dijkstra's algorithm

### Explanation:
Dijkstra's algorithm is a graph search algorithm that solves the single-source shortest path problem for a graph with non-negative edge path costs, producing the shortest path tree. This algorithm is often used in routing and as a subroutine in other graph algorithms.

Here are the basic steps:

1. **Initialization**: 
   - Set a tentative distance value for every node: set the initial node's distance to zero and all other nodes' distance to infinity. 
   - Set the initial node as the current node.
   - Mark all nodes as unvisited. Create a set of all the unvisited nodes.

2. **Main loop**:
   - For the current node, consider all of its neighbors and calculate their tentative distances through the current node. Compare the newly calculated tentative distance to the current assigned value and update the distance if the new value is smaller.
   - Once you have considered all of the neighbors of the current node, mark the current node as visited. A visited node will not be checked again.
   - Select the unvisited node with the smallest tentative distance, and set it as the new current node. Then go back to the previous step.

3. **Completion**:
   - The algorithm ends when every node has been visited. The algorithm has now constructed the shortest path tree from the source node to all other nodes.

### TypeScript Implementation:
```typescript
type Graph = {
    [key: string]: { [key: string]: number };
};

const dijkstra = (graph: Graph, start: string): { distances: { [key: string]: number }, previous: { [key: string]: string | null } } => {
    let distances: { [key: string]: number } = {};
    let previous: { [key: string]: string | null } = {};
    let unvisitedNodes = Object.keys(graph);

    for (let node of unvisitedNodes) {
        distances[node] = Infinity;
        previous[node] = null;
    }
    distances[start] = 0;

    while (unvisitedNodes.length !== 0) {
        unvisitedNodes.sort((a, b) => distances[a] - distances[b]);
        let currentNode = unvisitedNodes.shift() as string;

        for (let neighbor in graph[currentNode]) {
            let newDistance = distances[currentNode] + graph[currentNode][neighbor];

            if (newDistance < distances[neighbor]) {
                distances[neighbor] = newDistance;
                previous[neighbor] = currentNode;
            }
        }
    }
    return { distances, previous };
};

// Example Usage:
const exampleGraph: Graph = {
    A: { B: 1, D: 3 },
    B: { A: 1, D: 2, E: 5 },
    D: { A: 3, B: 2, E: 1 },
    E: { B: 5, D: 1 },
};

let result = dijkstra(exampleGraph, "A");
console.log(result.distances);
console.log(result.previous);
```

### Dry Run:
Let's dry run the algorithm using the `exampleGraph`:

1. **Initialization**: 
   - `distances = { A: 0, B: Infinity, D: Infinity, E: Infinity }`
   - `previous = { A: null, B: null, D: null, E: null }`
   - Current node = `A`
   - Unvisited nodes = `A, B, D, E`

2. **First Iteration**:
   - Current Node: `A`
   - Neighbors: `B` and `D`
     - For `B`: New distance = `0 + 1 = 1`, which is less than `Infinity`
       - `distances[B] = 1`, `previous[B] = A`
     - For `D`: New distance = `0 + 3 = 3`, which is less than `Infinity`
       - `distances[D] = 3`, `previous[D] = A`
   - Mark `A` as visited.
   - New current node = `B` (smallest distance among unvisited nodes)

3. **Second Iteration**:
   - Current Node: `B`
   - Neighbors: `A`, `D`, and `E`
     - For `D`: New distance = `1 + 2 = 3`, which is not less than current `3`
     - For `E`: New distance = `1 + 5 = 6`, which is less than `Infinity`
       - `distances[E] = 6`, `previous[E] = B`
   - Mark `B` as visited.
   - New current node = `D` (smallest distance among unvisited nodes)

4. **Third Iteration**:
   - Current Node: `D`
   - Neighbors: `A`, `B`, and `E`
     - For `E`: New distance = `3 + 1 = 4`, which is less than current `6`
       - `distances[E] = 4`, `previous[E] = D`
   - Mark `D` as visited.
   - New current node = `E` (last unvisited node)

5. **Fourth Iteration**:
   - Current Node: `E`
   - No updates since all neighbors have been visited.
   - Mark `E` as visited.

**Completion**:
`distances` = { A: 0, B: 1, D: 3, E: 4 }
`previous` = { A: null, B: 'A', D: 'A', E: 'D' }

The shortest path from A to E is `A -> B -> D -> E` with a distance of `4`.


### Time and Space Complexity:

Dijkstra's algorithm's time and space complexity are primarily influenced by the data structures used to implement it. The pseudocode provided earlier uses a basic array for the unvisited set and iterates over it to find the node with the smallest distance, which isn't the most efficient approach. Let's break down the complexities for the provided TypeScript implementation and then discuss how it can be optimized.

### For the provided TypeScript implementation:

**Time Complexity:**
1. Initializing `distances`, `previous`, and `unvisitedNodes` is \(O(V)\), where \(V\) is the number of vertices.
2. In the worst-case scenario, the `while` loop runs for every node, i.e., \(O(V)\).
3. Inside the `while` loop, the `sort` operation is \(O(V \log V)\).
4. Additionally, inside the `while` loop, we might go through all the edges of a node in the nested `for` loop, i.e., \(O(E)\) where \(E\) is the number of edges.

Multiplying these together, the worst-case time complexity is:
\[O(V \times (V \log V + E))\]
This can be approximated as \(O(V^2 \log V)\) in dense graphs where \(E \approx V^2\) and \(O(V^2)\) in sparse graphs.

**Space Complexity:**
1. The `distances` and `previous` maps have space complexity \(O(V)\).
2. The `unvisitedNodes` array also has space complexity \(O(V)\).

Summing these up, the overall space complexity is:
\[O(V)\]

### Optimized Version using Priority Queue:

You can optimize the time complexity by using a priority queue (or a binary heap) to manage the unvisited nodes. This would allow you to efficiently extract the node with the smallest tentative distance without having to sort the entire set of unvisited nodes each time.

With this optimization:

**Time Complexity:**
1. Initialization is still \(O(V)\).
2. The loop runs for every node and edge, i.e., \(O(V + E)\).
3. Extracting the minimum node from a priority queue is \(O(\log V)\).

Multiplying these together, the worst-case time complexity becomes:
\[O((V + E) \log V)\]

For a dense graph, this still reduces to \(O(V^2 \log V)\), but the constant factors are typically much better than the array-based approach. For sparse graphs, this is much faster, at \(O(V \log V)\).

**Space Complexity:**
The space complexity remains largely unchanged at \(O(V)\) for the data structures in the algorithm. However, if you include the priority queue's internal structures, it can go up slightly but remains within \(O(V)\) bounds.

In summary, while the naive version can be quite slow for large graphs, using a priority queue can significantly speed up Dijkstra's algorithm.

## Sorting

### Merge Sort

Merge Sort is a divide and conquer algorithm. There are 2 parts. Divide and then merge in right order.

- base case is single ordered item
- split into two lists
- recurse through 2 parts of list
- merge the two in correct order by checking each element consecutively

```python
def mergeSort(nums: List[int]) -> List[int]:

    l, r = 0, len(nums) - 1

    def divide(l: int, r: int):
        if left > right:
            return []

        if l == r:
            return [nums[left]] 

        mid = (r + l) // 2

        left = divide(l, mid)
        right = divide(mid+1, r)

        return merge(left, right)

    def merge(left_arr: List[int], right_arr: List[int]) -> List[int]:

        res = []
        i = j = 0

        # Merge while both arrays have elements
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                res.append(left_arr[i])
                i += 1
            else:
                res.append(right_arr[j])
                j += 1

        # Append any leftover elements
        res.extend(left_arr[i:])
        res.extend(right_arr[j:])
        return res

```

Merge sort on linked list

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            
        if not head or not head.next:
            return head

        slow = head
        fast = head

        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        # we split the linked list into 2
        mid = slow.next
        slow.next = None

        left = self.sortList(head)
        right = self.sortList(mid)

        return self.merge(left, right)

    def merge(self, left, right):
        
        dummy = ListNode(0)
        curr = dummy

        while left and right:
            if left.val <= right.val:
                curr.next = left
                left = left.next
            else:
                curr.next = right
                right = right.next
                
            curr = curr.next

        if not left:
            curr.next = right
        elif not right:
            curr.next = left

        return dummy.next

```

### Counting Sort
Counting Sort, which is an efficient sorting algorithm for non-negative integers within a limited range. Inefficient for large maxValue due to high space usage

Template
```
def countingSort(inputArray, maxValue):
    1. Initialize an array "count" of zeros with a size of (maxValue + 1).
    2. For each element "x" in inputArray:
       a. Increment count[x] by 1.

    3. Initialize an output array "sortedArray" of the same size as inputArray.
    4. Initialize a position variable "pos" to 0.
    5. For each index "i" from 0 to maxValue:
       a. While count[i] is greater than 0:
          i. Place the value "i" in sortedArray[pos].
          ii. Increment pos by 1.
          iii. Decrement count[i] by 1.

    6. Return sortedArray.
```

Example
```python
def countingSort(arr: List[Number]) -> List[Number]:

    maxValue = max(arr)  # Find the max value in arr
    count = [0 for _ in range(maxValue+1)] # Frequency array (index is the number, value is count)
    
    for i in range(len(arr)):
        count[arr[i]] += 1

    sortedArray = []
    for i in range(len(count)):
        sortedArray.extend([i] * count[i])

    return sortedArray
```

### Quick Sort

Quick Sort is a divide-and-conquer algorithm that works on the principle of choosing a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.

1. **Choose a Pivot**: Select an element from the array as the pivot. This can be done in various ways, such as choosing the first element, the last element, the middle element, or even a random element.

2. **Partitioning**: Reorder the array so that all elements with values less than the pivot come before the pivot, while all elements with values greater than the pivot come after it. After this partitioning, the pivot is in its final position.

3. **Recursive Sorting**: Recursively apply the above steps to the sub-array of elements with smaller values and the sub-array of elements with greater values.

4. **Base Case**: The recursion base case is an array with zero or one element, which doesn't need to be sorted.

Now, here's an example of Quick Sort implemented in TypeScript:

```typescript
function quickSort(arr: number[], low: number, high: number): void {
  if (low < high) {
    // Partition the array and get the pivot index
    let pi = partition(arr, low, high);

    // Recursively sort the elements before partition and after partition
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

function partition(arr: number[], low: number, high: number): number {
  // Choose the rightmost element as pivot
  let pivot = arr[high];

  // Pointer for greater element
  let i = low - 1;

  // Traverse through all elements
  // compare each element with pivot
  for (let j = low; j < high; j++) {
    if (arr[j] < pivot) {
      // If element smaller than pivot is found
      // swap it with the greater element pointed by i
      i++;

      // Swapping element at i with element at j
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  // Swap the pivot element with the greater element specified by i
  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];

  // Return the position from where partition is done
  return i + 1;
}

// Helper function to initiate QuickSort
function quickSortHelper(arr: number[]): number[] {
  quickSort(arr, 0, arr.length - 1);
  return arr;
}

// Example usage:
const arr = [10, 7, 8, 9, 1, 5];
console.log('Original Array:', arr);
const sortedArray = quickSortHelper(arr);
console.log('Sorted Array:', sortedArray);
```



# Bit Manipulation

Bits are base 2, hexadecimal is base 16 where numbers 10 and above are represented by letters up to 16

The index represents the power.
```
a = "1010"
```
This is

```
1x2^3 + 0x2^2 + 1x2^1 + 0x2^0 = 10
```

Two's complement is used to represent negative numbers (simplifies computer arithmetic by eliminating the need for separate addition/subtraction circuits and handling the sign bit): 
- Invert all bits (1s complement)
- Add 1 to the result

Example for -10:
```
Original:     1010 (10)
Invert:       0101
Add 1:        0110 (-10 in two's complement)
```

When adding twos compliment and non twos compliment form:
 - All bits (including the sign bit) take part in the binary addition in two’s complement.
 - discard any extra carry beyond the nth bit because you are working with a fixed size

to add 5 and -3, we must get -3 by converting 3 to -3 using twos compliment then
 e.g. 
 ```
0101 (+5) + 1101 (-3) = 0010 (2)
```

```python
# AND (&): 1 if both bits are 1
    1010
  & 1100
  = 1000

# OR (|): 1 if either bit is 1
    1010
  | 1100
  = 1110

# XOR (^): 1 if bits are different
    1010
  ^ 1100
  = 0110

# NOT (~): Inverts all bits
  ~ 1010
  = 0101

# Left Shift (<<): Shifts bits left, adds 0s
  1010 << 1 = 10100

# Right Shift (>>): Shifts bits right
  1010 >> 1 = 101
```
Common Bit Manipulation Tricks:
```python
# Check if power of 2
n & (n-1) == 0

# Get last set bit
n & -n

# Count set bits (1s)
bin(n).count('1')

# Check if ith bit is set
n & (1 << i)

# Set ith bit
n | (1 << i)

# Clear ith bit
n & ~(1 << i)

# Toggle ith bit
n ^ (1 << i)
```