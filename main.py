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







sol = Solution()
print(sol.longestValidParentheses_3('()(())'))