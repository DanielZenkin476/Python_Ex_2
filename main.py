from bisect import bisect_right
from distutils.command.check import check
from inspect import stack
from itertools import filterfalse



class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        return bisect_right(range(x+1),0,key = lambda q:q*q-x)-1

    def mySqrt_2(self,x):
        if x == 0:
            return 0
        if x<4:
            return 1
        left = 0
        right = x
        while(left<right):
            mid = (left+right)//2
            mult = mid*mid
            mult_plus = (mid+1)*(mid+1)
            if mult == x:
                return mid
            if mult<x:
                if mult_plus>x:
                    return mid
                left= mid+1
            if mult>x:
                right = mid-1
        return right

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n<=1:# only one element or none
            return
        if n==2:# [x,y] -> [y,x]
            nums.reverse()
            return
        i = n-2
        while(i>=0 and nums[i]>=nums[i+1]):# check if aray is in ascending order -if so reverse
            i-=1
        if i ==-1:
            nums.reverse()
            return
        j = n-1
        while nums[j]<=nums[i]:
            j-=1
        nums[i],nums[j] = nums[j],nums[i]
        nums[i+1:] = reversed(nums[i+1:])

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == "": return 0
        s_lst = []
        curr = 0
        max_len = 0
        id = 0
        flag = False
        n = len(s)
        sav_curr = 0
        while(id<n):
            char = s[id]
            if char == '(':
                if not s_lst:
                    flag = True# lst was not empty
                    sav_curr = curr
                    curr = 0
                s_lst.append(char)
            elif char ==')':
                if s_lst and  s_lst.pop() == '(':
                    curr+=2
                    if not s_lst:
                        max_len = max(max_len, curr+sav_curr)
                    else : max_len = max(max_len, curr)
                else: #invalid
                    flag = False
                    curr= 0
                    s_lst = []
            id+=1
        if flag and not s_lst:
            return max_len+sav_curr
        return max_len

    def longestValidParentheses_2(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_len = 0

        left, right = 0, 0
        for i in range(len(s)):# go from left to right
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:# substring valid
                max_len = max(max_len, left * 2)
            elif right > left:# invalid case
                left = right = 0

        left, right = 0, 0
        for i in range(len(s) - 1, -1, -1):# go from right to left
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:# substring valid
                max_len = max(max_len, left * 2)
            elif left > right:# invalid case
                left = right = 0
        return max_len

    def longestValidParentheses_3(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_len =0
        char_lst = [-1]
        for i in range(len(s)):
            if s[i]== '(':
                char_lst.append(i)
            else:
                char_lst.pop()
                if not char_lst:
                    char_lst.append(i)
                else:
                    max_len = max(max_len,i-char_lst[-1])
        return max_len

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        while (left<=right):
            mid = (left+right)//2
            if nums[mid]==target:
                return mid
            if nums[left]<= nums[mid]:
                if nums[left]<= target < nums[mid]:
                    right = mid-1
                else :
                    left = mid+1
            else:
                if nums[mid]<target<= nums[right]:
                    left = mid + 1
                else:
                    right = mid-1
        return -1

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n=len(nums)
        left = 0
        right = n- 1
        while (left <= right):
            mid = (left + right) // 2
            if nums[mid] == target:
                s = end = mid
                while(s>0 and nums[s-1]==target):
                    s-=1
                while (end<n-1 and nums[end+1] == target):
                    end += 1
                return [s,end]
            if nums[mid]< target:
                left = mid+1
            else:
                right = mid-1
        return [-1,-1]

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        res = 1.0
        if n>0:
            res = 1.0
            while(n>0):
                res*=x
                n-=1
        else:
            n=-n
            res = 1.0
            while(n>0):
                res= res/x
                n-=1
        return res

    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        if n ==1: return matrix
        check = [[False for a in range(n)] for b in range(n)]
        x = 0
        y = 0
        while(x<n/2):
            y=0
            while(y< n/2):
                if check[x][y]== False:
                    check[x][y]= True
                    check[y][n-1-x]= True
                    check[n-1-y][n-1-x]=True
                    check[n-1-y][x] = True
                    self.switch_image(matrix,x,y,y,n-1-x)
                    self.switch_image(matrix,x,y,n-1-x,n-1-y)
                    self.switch_image(matrix,x,y,n-1-y,x)
                else: pass
                y+=1
            x+=1

    def switch_image(self,matrix,a,b,c,d):
        temp = matrix[a][b]
        matrix[a][b] =matrix[c][d]
        matrix[c][d] = temp

    def deleteDuplicates(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not head:
            return head
        node = head
        while node and node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n ==1:
            return 1
        elif n==2 : return 2
        else: return self.climbStairs(n-1)+self.climbStairs(n-2)

    def climbStairs_2(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n ==0: return 0
        if n ==1 : return 1
        if n==2: return 2
        a =1
        b = 2
        for i in range(3,n+1):
            temp = b
            b = a+b
            a = temp
        return b

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        if n ==0 :
            return
        m = len(matrix[0])
        if m ==0:
            return
        rows_z =set()
        col_z =set()
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    rows_z.add(i)
                    col_z.add(j)
        for row in rows_z:
            for j in range(m):
                matrix[row][j]=0
        for col in col_z:
            for i in range(n):
                matrix[i][col]=0

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        if m == 0:
            return False
        if target < matrix[0][0] or target>matrix[n-1][m-1]:
            return False
        start = 0
        end = n-1
        while (start <= end):
            row = (start + end) // 2
            if matrix[row][0]<=target<= matrix[row][m-1]: break
            elif target> matrix[row][m-1]:
                start = row + 1
            else:
                end = row-1
        start = 0
        end = m-1
        while(start<=end):
            mid = (start+end)//2
            if matrix[row][mid] == target: return True
            elif matrix[row][mid] > target:
                end = mid-1
            else:
                start = mid+1
        return False

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for item in nums:
            for i in range(len(res)):
                res.append(res[i]+[item])
        return res

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        if n==0:
            return
        if m==0:
            for i in range(n):
                nums1[i]=nums2[i]
            return
        nums1=nums1[m:]+nums1[:m]
        i = n
        j= 0
        id = 0
        while(i<m+n and j<n):
            if nums1[i]<nums2[j]:
                nums1[id] = nums1[i]
                i+=1
            else:
                nums1[id] = nums2[j]
                j += 1
            id+=1
        while(i<m+n):
            nums1[id] = nums1[i]
            i += 1
            id += 1
        while (j<n):
            nums1[id] = nums2[j]
            j += 1
            id += 1

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        n = 9
        for i in range(n):# goes over rows
            bool_arr = [False for a in range(9)]
            for j in range(n):
                if board[i][j]!= '.':
                    if bool_arr[int(board[i][j])-1]:
                        return False
                    else:
                        bool_arr[int(board[i][j])-1] = True
        for i in range(n):# goes over col
            bool_arr = [False for a in range(n)]
            for j in range(n):
                if board[j][i]!= '.':
                    if bool_arr[int(board[j][i])-1]:
                        return False
                    else:
                        bool_arr[int(board[j][i])-1] = True
        for x in [0,3,6]:
            for y in [0,3,6]:
                if not self.check_sq(board,x,y): return False

        return True

    def check_sq(self, board,startx,starty):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        bool_arr = [False for a in range(9)]
        for i in range(startx,startx+3):
            for j in range(starty,starty+3):
                if board[i][j] != '.':
                    if bool_arr[int(board[i][j])-1]:
                        return False
                    else:
                        bool_arr[int(board[i][j])-1] = True
        return True

    def isValidSudoku_2(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        rows,cols = len(board),len(board[0])
        for r in range(rows):
            r_map = set()
            for c in range(cols):
                if board[r][c] in r_map and board[r][c]!= ".":
                    return False
                r_map.add(board[r][c])

        for c in range(cols):
            c_map = set()
            for r in range(rows):
                if board[r][c] in c_map and board[r][c]!= ".":
                    return False
                c_map.add(board[r][c])

        for i in range(0,9,3):
            for j in range(0,9,3):
                s_map = set()
                for r in range(3):
                    for c in range(3):
                        if board[i+r][j+c] in s_map and board[i+r][j+c] != ".":
                            return False
                        s_map.add(board[i+r][j+c])
        return True

    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n ==1:return "1"
        last_str = self.countAndSay(n-1)
        res = ""
        i =0
        while i <len(last_str):
            curr_num= last_str[i]
            curr_count = 1
            i+=1
            while( i < len(last_str) and curr_num== last_str[i]):
                i+=1
                curr_count+=1
            res+=str(curr_count)+curr_num
        return res

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        poss_res =[]
        candidates = sorted(candidates)
        self.rec_sum(candidates,target,res = [],final_res=poss_res)
        return poss_res

    def rec_sum(self,candidates,target,res,final_res):
        if target== 0:
            return
        if target < 0:
            return
        else:
            for num in candidates:
                if num ==target:
                    sol = sorted([num] + res)
                    if sol not in final_res: final_res.append(sol)
                elif num<target:
                    self.rec_sum(candidates,target-num,res+[num],final_res)
        return

    def rec_sum_sorted(self,candidates,target,res,final_res):
        if target== 0:
            return
        if target < 0:
            return
        else:
            for i in range(len(candidates)):
                if candidates[i]> target:
                    break
                elif candidates[i] ==target:
                    sol = sorted([candidates[i]] + res)
                    if sol not in final_res: final_res.append(sol)
                else:
                    self.rec_sum(candidates,target-candidates[i],res+[candidates[i]],final_res)
        return

    def inorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        pre_order = []
        self.preorder_rec(root,pre_order)
        in_order = []
        self.inorder_rec(root,in_order)
        post_order = []
        self.postoder_rec(root,post_order)

        return in_order

    def inorder_rec(self,node,lst):
        if node is None:
            return
        else:
            self.inorder_rec(node.left, lst)
            lst.append(node.val)
            self.inorder_rec(node.right, lst)

    def postoder_rec(self,node,lst):
        if node is None:
            return
        else:
            self.inorder_rec(node.left, lst)
            self.inorder_rec(node.right, lst)
            lst.append(node.val)

    def preorder_rec(self,node,lst):
        if node is None:
            return
        else:
            lst.append(node.val)
            self.inorder_rec(node.left, lst)
            self.inorder_rec(node.right, lst)

    def isSameTree(self, p, q):
        """
        :type p: Optional[TreeNode]
        :type q: Optional[TreeNode]
        :rtype: bool
        """
        if q is None and p is None:
            return True
        elif q is None or p is None:
            return False
        else:
            if q.val != p.val: return False
            else:
                return self.isSameTree(q.left,p.left) and self.isSameTree(q.right,p.right)

    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        left = root.left
        right = root.right
        def symmetry(r,l):
            if r is None and l is None:
                return True
            elif l is None or r is None:
                return False
            else:
                if l.val!= r.val: return False
                return symmetry(r.right,l.left) and symmetry(r.left,l.right)
        return symmetry(right,left)

    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root is None:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        nxt =1
        curr = 0
        while nxt<n:
            if nums[curr]== nums[nxt]:#curr == next
                nxt+=1# move nxt forward
                while nxt<n and nums[nxt]== nums[curr]:# next one is atleats third - need to push him back
                    nums[nxt:n]=nums[nxt+1:n]+[nums[curr]]# move list back and insert curr in the last place (n is also k)
                    n-=1
                curr = nxt
                nxt+=1
            else:
                curr = nxt
                nxt +=1
        return n

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board or not board[0]:
            return False
        rows= len(board)
        cols = len(board[0])
        for row in range(rows):
            for col in range(cols):
                if self.seq_check(board,word,0,row,col,[]): return True
        return False

    def seq_check(self,board,word,id,r,c,id_list):
        if r<0 or c<0 or r>= len(board) or c>=len(board[0]) or (r,c)in id_list:
            return False
        if id == len(word):
            return True
        if board[r][c] == word[id]:
            id_list.append((r,c))
            p1 = self.seq_check(board,word,id+1,r-1,c,id_list)
            p2 = self.seq_check(board,word,id+1,r+1,c,id_list)
            p3= self.seq_check(board,word,id+1,r,c-1,id_list)
            p4= self.seq_check(board,word,id+1,r,c+1,id_list)
            if p1 or p2 or p3 or p4: return True
        return False

    def sortColors(self, nums):
        """
        :type nums: List[int], of 0 ,1 ,2 only
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        count = [0,0,0] # first for 0, second for 1, third for 2
        for num in nums:
            count[num]+=1
        c_id = 0
        i = 0
        for i in range(len(nums)):
            while(count[c_id]==0):
                c_id+=1
            nums[i]= c_id
            count[c_id]-=1
        print(nums)

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        if k>n or k ==0 or n ==0 :
            return []
        res = []
        lst = []
        for i in range(1,k+1):
            lst.append(i)
        res.append(lst)
        id = k-1
        while lst[id]<n and id >=0:
            lst = list(lst)
            lst[id]+=1
            res.append(lst)
            if lst[id]==n:
                if id ==0 :break
                else:
                    id-=1
                    n-=1
        return res

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        if k>n or k ==0 or n ==0 :
            return []
        res = []
        lst = []
        for i in range(1,k+1):
            lst.append(i)
        res.append(lst)
        id = k-1
        while lst[id]<n and id >=0:
            lst = list(lst)
            lst[id]+=1
            res.append(lst)
            if lst[id]==n:
                if id ==0 :break
                else:
                    id-=1
                    n-=1
        return res

    def rotateRight(self, head, k):
        """
        :type head: Optional[ListNode]
        :type k: int
        :rtype: Optional[ListNode]
        """
        if head is None or head.next is None or k ==0:
            return head
        l=0
        loop_flag = True
        break_flag = True
        node = head
        new_start = -1
        while loop_flag or break_flag:
            if node.next is None:
                l+=1
                node.next = head
                loop_flag = False
                while (l<=k):
                    k-=l
                new_start = l-k
                if new_start== 0:
                    return head
                l= 0
            else:
                l += 1
                if l == new_start+1:
                    temp = node.next
                    node.next = None
                    node = temp
                    break_flag = False
                else:
                    node = node.next
        return temp


### need to check ###


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


a =ListNode(5,None)
b =ListNode(4,a)
c =ListNode(3,b)
d =ListNode(2,c)
h =ListNode(1,d)



sol = Solution()
new_h =sol.rotateRight(h,6)
while(new_h):
    print(new_h.val)
    new_h=new_h.next