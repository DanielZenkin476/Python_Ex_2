from bisect import bisect_right
from distutils.command.check import check
from inspect import stack


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
















#Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next





sol = Solution()
print(sol.rotate(matrix = [[1,2,3],[4,5,6],[7,8,9]]))